"""
Local web viewer for Grad-CAM uncertainty analyses (Feature 4 of the
uncertainty toolkit).

Serves the output folder produced by ``donkeycar.parts.gradcam_uncertainty``
together with a small self-contained web page (no internet required) that
provides:

  * the camera frame with the uncertainty-map overlay (or mean attention,
    or the plain frame),
  * a confidence-over-time graph for the whole drive with the current
    position marked, click-to-jump,
  * video-like playback (play/pause at a configurable 2-10 fps, scrub bar,
    arrow-key stepping, optional "analysed frames only" mode).

Frames without a precomputed uncertainty map simply show the camera frame
(from the analysis dir if it was created with ``--export-frames``, else
straight from the tub's images/ folder when the tub is available locally).

Usage:
    python -m donkeycar.parts.uncertainty_viewer --analysis <dir> \
        [--tub <tub_dir>] [--port 8890] [--host 127.0.0.1]

Then open http://localhost:8890 in a browser.
"""
import argparse
import json
import logging
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger(__name__)

HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'uncertainty_viewer.html')


class ViewerHandler(SimpleHTTPRequestHandler):
    """Serves the analysis directory, the viewer page at '/', and (when a
    tub is available) raw camera frames at '/tub/<filename>'."""

    # set once in main(); shared by all request threads (read-only)
    tub_images_dir = None

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            return self._send_file(HTML_PATH, 'text/html; charset=utf-8')

        if self.path.startswith('/tub/'):
            if not self.tub_images_dir:
                return self.send_error(404, 'tub not available')
            # basename() blocks any path traversal
            name = os.path.basename(self.path[len('/tub/'):])
            file_path = os.path.join(self.tub_images_dir, name)
            if not os.path.isfile(file_path):
                return self.send_error(404)
            ctype = 'image/png' if name.lower().endswith('.png') \
                else 'image/jpeg'
            return self._send_file(file_path, ctype)

        # everything else (data.json, images/, frames/) comes from the
        # analysis directory via the base handler
        return super().do_GET()

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

    def log_message(self, fmt, *args):
        logger.debug(fmt % args)


def main(args=None):
    parser = argparse.ArgumentParser(
        prog='uncertainty_viewer',
        description='Web viewer for Grad-CAM uncertainty analyses.')
    parser.add_argument('--analysis', required=True,
                        help='analysis dir produced by gradcam_uncertainty '
                             '(must contain data.json)')
    parser.add_argument('--tub', default=None,
                        help='tub dir for camera frames of non-analysed '
                             'frames (default: the tub path recorded in '
                             'data.json, if it exists on this machine)')
    parser.add_argument('--port', type=int, default=8890)
    parser.add_argument('--host', default='127.0.0.1',
                        help='bind address (use 0.0.0.0 to allow access '
                             'from other machines)')
    parsed = parser.parse_args(args)

    analysis_dir = os.path.abspath(os.path.expanduser(parsed.analysis))
    data_json = os.path.join(analysis_dir, 'data.json')
    if not os.path.isfile(data_json):
        raise SystemExit(f'No data.json in {analysis_dir} -- run '
                         f'donkeycar.parts.gradcam_uncertainty first.')

    # Resolve the tub images dir for non-analysed frames: --tub wins, else
    # the tub path stored in data.json (which may not exist on this machine
    # -- that's fine, exported frames or a placeholder cover it).
    tub_dir = parsed.tub
    if tub_dir is None:
        with open(data_json) as f:
            tub_dir = json.load(f).get('tub')
    tub_images = None
    if tub_dir:
        candidate = os.path.join(os.path.expanduser(tub_dir), 'images')
        if os.path.isdir(candidate):
            tub_images = os.path.abspath(candidate)
    ViewerHandler.tub_images_dir = tub_images
    logger.info(f'Analysis dir : {analysis_dir}')
    if tub_images:
        logger.info(f'Tub images   : {tub_images}')
    else:
        logger.info('Tub images   : not available (exported frames / '
                    'placeholder used for non-analysed frames)')

    handler = partial(ViewerHandler, directory=analysis_dir)
    server = ThreadingHTTPServer((parsed.host, parsed.port), handler)
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
