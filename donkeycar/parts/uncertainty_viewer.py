"""
Local web viewer for Grad-CAM uncertainty analyses (Feature 4 of the
uncertainty toolkit) -- and, when no analysis is given yet, a small local
GUI ("launcher") for running one, so you never have to type the full
``python -m donkeycar.parts.gradcam_uncertainty --tub ... --model ...``
command line by hand.

Serves the output folder produced by ``donkeycar.parts.gradcam_uncertainty``
together with a small self-contained web page (no internet required) that
provides:

  * the camera frame with the uncertainty-map overlay (or mean attention,
    novelty, saliency, or the plain frame),
  * a confidence-over-time graph for the whole drive with the current
    position marked, click-to-jump,
  * video-like playback (play/pause at a configurable 2-10 fps, scrub bar,
    arrow-key stepping, optional "analysed frames only" mode).

Frames without a precomputed uncertainty map simply show the camera frame
(from the analysis dir if it was created with ``--export-frames``, else
straight from the tub's images/ folder when the tub is available locally).

**Launcher mode**: if ``--analysis`` is omitted (or points at a directory
with no ``data.json`` yet), the server shows a form instead of the viewer:
pick a tub and model (autodiscovered from ``data/`` and ``models/`` in the
current directory, or typed by hand), choose which frames to analyse, and
click Run. The analysis runs in this same process on a background thread
(reusing ``gradcam_uncertainty.analyze_tub`` directly -- no subprocess), the
page polls progress, and once done the server starts serving the viewer for
the new output directory automatically -- no second command.

Usage:
    python -m donkeycar.parts.uncertainty_viewer [--analysis <dir>] \
        [--tub <tub_dir>] [--config <config.py>] [--port 8890] [--host 127.0.0.1]

Then open http://localhost:8890 in a browser.
"""
import argparse
import json
import logging
import os
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

logger = logging.getLogger(__name__)

HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'uncertainty_viewer.html')
LAUNCHER_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'analysis_launcher.html')


class LauncherState:
    """
    Owns the background analysis thread and its progress state. One instance
    per server process, shared (read via a lock) across request-handling
    threads.
    """

    def __init__(self, cfg, cwd):
        self.cfg = cfg
        self.cwd = cwd
        self._lock = threading.Lock()
        self._progress = {'running': False, 'done': False, 'error': None,
                          'stage': None, 'current': 0, 'total': 0,
                          'out_dir': None}

    def progress_snapshot(self):
        with self._lock:
            return dict(self._progress)

    def _set(self, **kwargs):
        with self._lock:
            self._progress.update(kwargs)

    def scan_defaults(self):
        """Best-effort discovery of tub dirs (contain manifest.json) and
        model .h5 files under the current directory, for the form's
        suggestion lists. Returns {} entries if nothing found -- the form
        fields are free text either way."""
        tubs = []
        data_dir = os.path.join(self.cwd, 'data')
        if os.path.isfile(os.path.join(data_dir, 'manifest.json')):
            tubs.append('data')
        if os.path.isdir(data_dir):
            for name in sorted(os.listdir(data_dir)):
                p = os.path.join(data_dir, name)
                if os.path.isdir(p) and os.path.isfile(
                        os.path.join(p, 'manifest.json')):
                    tubs.append(os.path.join('data', name))

        models = []
        models_dir = os.path.join(self.cwd, 'models')
        if os.path.isdir(models_dir):
            for name in sorted(os.listdir(models_dir)):
                if name.lower().endswith('.h5'):
                    models.append(os.path.join('models', name))

        return {'tubs': tubs, 'models': models}

    def tub_record_count(self, tub_path):
        """
        Best-effort total record count for a tub, read from its
        ``manifest.json`` (a few small JSON-per-line records -- no need to
        load the whole catalog dataset just to size the form's validation).
        Returns None if it can't be determined (bad path, unexpected
        manifest format).
        """
        manifest_path = os.path.join(
            os.path.expanduser(tub_path), 'manifest.json')
        if not os.path.isfile(manifest_path):
            return None
        try:
            with open(manifest_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    if isinstance(d, dict) and 'current_index' in d:
                        n_deleted = len(d.get('deleted_indexes', []))
                        return max(0, d['current_index'] - n_deleted)
        except Exception as e:
            logger.debug(f'tub_record_count({tub_path}) failed: {e}')
        return None

    def start(self, form):
        with self._lock:
            if self._progress['running']:
                raise RuntimeError('An analysis is already running.')
            self._progress = {'running': True, 'done': False, 'error': None,
                              'stage': 'starting', 'current': 0, 'total': 0,
                              'out_dir': None}
        thread = threading.Thread(target=self._run, args=(form,), daemon=True)
        thread.start()

    def _run(self, form):
        try:
            tub = form['tub']
            model = form['model']
            if not tub or not model:
                raise ValueError('Tub and model paths are required.')
            mode = form.get('mode', 'top_k')
            out_dir = form.get('out') or os.path.join(
                os.path.expanduser(tub), 'gradcam_analysis')

            def cb(stage, current, total):
                self._set(stage=stage, current=current, total=total)

            from donkeycar.parts.mc_calibrate import (default_calib_path,
                                                      calibrate_from_tub)
            calib_path = default_calib_path(os.path.expanduser(model))
            # Auto-calibrate models that have never been calibrated, so a
            # first-time user gets confidence/novelty %% without having to
            # know `mc_calibrate` exists as a separate command. Opt-out via
            # the form checkbox; a calibration failure here degrades to the
            # pre-existing behaviour (variance-only) rather than blocking the
            # analysis the user actually asked for.
            if (form.get('auto_calibrate', True)
                    and not os.path.exists(calib_path)):
                logger.info(f'No calibration at {calib_path}; '
                            f'auto-calibrating before analysis.')
                try:
                    calibrate_from_tub(self.cfg, [tub], model,
                                       progress_callback=cb)
                except Exception as e:
                    logger.warning(f'Auto-calibration failed ({e}); '
                                   f'continuing without it -- confidence/'
                                   f'novelty %% will be unavailable.')

            from donkeycar.parts.gradcam_uncertainty import analyze_tub
            analyze_tub(
                self.cfg, tub, model, out_dir,
                num_passes=getattr(self.cfg, 'XAI_CONFIDENCE_PASSES', 15),
                alpha=getattr(self.cfg, 'XAI_CONFIDENCE_ALPHA', 0.2),
                ig_steps=getattr(self.cfg, 'XAI_IG_STEPS', 32),
                top_k=int(form['value']) if mode == 'top_k'
                      and form.get('value') else 50,
                percentile=float(form['value']) if mode == 'percentile'
                          and form.get('value') else None,
                analyze_all=(mode == 'all'),
                limit=int(form['limit']) if form.get('limit') else None,
                start=int(form['start']) if form.get('start') else None,
                export_frames=bool(form.get('export_frames')),
                progress_callback=cb)

            # Point the viewer at the freshly-produced analysis.
            ViewerHandler.active_dir = os.path.abspath(out_dir)
            candidate = os.path.join(os.path.expanduser(tub), 'images')
            ViewerHandler.tub_images_dir = \
                os.path.abspath(candidate) if os.path.isdir(candidate) else None

            self._set(running=False, done=True, out_dir=out_dir)
        except Exception as e:
            logger.exception('Analysis failed')
            self._set(running=False, done=True, error=str(e))


class ViewerHandler(SimpleHTTPRequestHandler):
    """
    Serves either the analysis viewer (if ``active_dir`` has a ready
    ``data.json``) or the launcher form (otherwise), plus a small JSON API
    for the launcher (``/api/scan``, ``/api/run``, ``/api/progress``) and raw
    camera frames at ``/tub/<filename>`` when a tub is available.

    ``active_dir``/``tub_images_dir``/``launcher_state`` are mutable class
    attributes rather than constructor args, because ``http.server`` creates
    a fresh handler instance per request -- setting them here lets the
    *directory being served* change at runtime (e.g. once a launcher-mode
    analysis finishes) without restarting the server.
    """

    tub_images_dir = None
    active_dir = None
    launcher_state = None

    def __init__(self, *args, **kwargs):
        directory = ViewerHandler.active_dir or os.getcwd()
        super().__init__(*args, directory=directory, **kwargs)

    def _analysis_ready(self):
        return (ViewerHandler.active_dir is not None
                and os.path.isfile(
                    os.path.join(ViewerHandler.active_dir, 'data.json')))

    def do_GET(self):
        if self.path.startswith('/') and '?' in self.path:
            path, query = self.path.split('?', 1)
        else:
            path, query = self.path, ''

        if path in ('/', '/index.html'):
            force_launcher = 'new=1' in query
            if not force_launcher and self._analysis_ready():
                return self._send_file(HTML_PATH, 'text/html; charset=utf-8')
            if ViewerHandler.launcher_state is not None:
                return self._send_file(LAUNCHER_HTML_PATH,
                                       'text/html; charset=utf-8')
            return self.send_error(
                404, 'No analysis available and launcher mode is off '
                     '(pass --analysis, or run without it to enable the '
                     'launcher).')

        if path == '/api/scan':
            if ViewerHandler.launcher_state is None:
                return self.send_error(404)
            return self._send_json(ViewerHandler.launcher_state.scan_defaults())

        if path == '/api/progress':
            if ViewerHandler.launcher_state is None:
                return self._send_json({})
            return self._send_json(
                ViewerHandler.launcher_state.progress_snapshot())

        if path == '/api/tub_info':
            if ViewerHandler.launcher_state is None:
                return self.send_error(404)
            tub = parse_qs(query).get('tub', [''])[0]
            if not tub:
                return self._send_json({'error': 'no tub given'}, status=400)
            n = ViewerHandler.launcher_state.tub_record_count(tub)
            if n is None:
                return self._send_json(
                    {'error': f'could not read manifest.json for {tub}'},
                    status=404)
            return self._send_json({'n_records': n})

        if path.startswith('/tub/'):
            if not self.tub_images_dir:
                return self.send_error(404, 'tub not available')
            # basename() blocks any path traversal
            name = os.path.basename(path[len('/tub/'):])
            file_path = os.path.join(self.tub_images_dir, name)
            if not os.path.isfile(file_path):
                return self.send_error(404)
            ctype = 'image/png' if name.lower().endswith('.png') \
                else 'image/jpeg'
            return self._send_file(file_path, ctype)

        # everything else (data.json, images/, frames/) comes from
        # active_dir via the base handler
        return super().do_GET()

    def do_POST(self):
        if self.path == '/api/run':
            if ViewerHandler.launcher_state is None:
                return self.send_error(404)
            length = int(self.headers.get('Content-Length', 0))
            try:
                form = json.loads(self.rfile.read(length)) if length else {}
                ViewerHandler.launcher_state.start(form)
                return self._send_json({'ok': True})
            except Exception as e:
                return self._send_json({'ok': False, 'error': str(e)},
                                       status=400)
        return self.send_error(404)

    def _send_file(self, path, ctype):
        try:
            with open(path, 'rb') as f:
                body = f.read()
        except OSError:
            return self.send_error(404)
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        logger.debug(fmt % args)


def _load_config(config_path):
    import donkeycar as dk
    if config_path is None and not os.path.exists('config.py'):
        config_path = os.path.join(os.path.dirname(dk.__file__),
                                   'templates', 'cfg_complete.py')
        logger.warning(f'No ./config.py; using bundled defaults for the '
                       f'launcher.')
    return dk.load_config(config_path)


def main(args=None):
    parser = argparse.ArgumentParser(
        prog='uncertainty_viewer',
        description='Web viewer (and analysis launcher) for Grad-CAM '
                    'uncertainty analyses.')
    parser.add_argument('--analysis', default=None,
                        help='analysis dir produced by gradcam_uncertainty '
                             '(must contain data.json). Omit to start in '
                             'launcher mode and run a new analysis from the '
                             'browser.')
    parser.add_argument('--tub', default=None,
                        help='tub dir for camera frames of non-analysed '
                             'frames (default: the tub path recorded in '
                             'data.json, if it exists on this machine)')
    parser.add_argument('--config', default=None,
                        help='path to config.py, used by launcher mode when '
                             'starting a new analysis (defaults to '
                             './config.py, falling back to bundled defaults)')
    parser.add_argument('--port', type=int, default=8890)
    parser.add_argument('--host', default='127.0.0.1',
                        help='bind address (use 0.0.0.0 to allow access '
                             'from other machines)')
    parsed = parser.parse_args(args)

    cwd = os.getcwd()
    cfg = _load_config(parsed.config)
    ViewerHandler.launcher_state = LauncherState(cfg, cwd)

    if parsed.analysis:
        analysis_dir = os.path.abspath(os.path.expanduser(parsed.analysis))
        data_json = os.path.join(analysis_dir, 'data.json')
        if os.path.isfile(data_json):
            ViewerHandler.active_dir = analysis_dir
            tub_dir = parsed.tub
            if tub_dir is None:
                with open(data_json) as f:
                    tub_dir = json.load(f).get('tub')
            if tub_dir:
                candidate = os.path.join(os.path.expanduser(tub_dir), 'images')
                if os.path.isdir(candidate):
                    ViewerHandler.tub_images_dir = os.path.abspath(candidate)
        else:
            logger.warning(f'{analysis_dir} has no data.json yet; starting '
                           f'in launcher mode instead.')

    logger.info(f'Analysis dir : {ViewerHandler.active_dir or "(none yet -- launcher mode)"}')
    if ViewerHandler.tub_images_dir:
        logger.info(f'Tub images   : {ViewerHandler.tub_images_dir}')

    server = ThreadingHTTPServer((parsed.host, parsed.port), ViewerHandler)
    shown_host = 'localhost' if parsed.host in ('127.0.0.1', '0.0.0.0') \
        else parsed.host
    print(f'\nUncertainty viewer running at '
          f'http://{shown_host}:{parsed.port}  (Ctrl+C to stop)\n')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
