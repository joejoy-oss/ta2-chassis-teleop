"""RTC-only registration and the documentation page against the offline HTTP server."""
import json
from pathlib import Path
import subprocess
import unittest
from urllib.request import urlopen

from test_api_modules import _ModuleServer

ROOT = Path(__file__).resolve().parents[1]
REMOVED = ('grasp_v3_step13000', 'grasp_v3_step10000', 'grasp_v3_step5000', 'sandbag_v2_step4999')


class APIDocsTests(_ModuleServer, unittest.TestCase):
    def test_docs_served_with_all_interfaces_and_main_link(self):
        with urlopen(self.url + '/api-docs') as response:
            self.assertEqual(response.status, 200)
            self.assertIn('text/html', response.headers['Content-Type'])
            html = response.read().decode()
        self.assertEqual(html, (ROOT / 'api_docs.html').read_text())
        for path in ('/health', '/state', '/capabilities', '/models', '/grippers', '/preflight',
                     '/actions', '/actions/<id>', '/actions/<id>/cancel', '/stop', '/estop'):
            self.assertIn("path:'" + path + "'", html)
        for action in self.client.capabilities()['actions']:
            self.assertIn("id:'" + action + "'", html)
        self.assertIn('会动机器人', html)
        self.assertNotIn('confirm(', html)
        self.assertIn("const API='/api/v1'", html)
        self.assertIn("if(!path.startsWith(API+'/'))", html)
        with urlopen(self.url + '/') as response:
            main = response.read().decode()
        self.assertIn('<a href="/api-docs" target="_blank" rel="noopener">接口文档 / 测试</a>', main)

    def test_rtc_catalog_and_removed_ids_rejected(self):
        models = self.client.models()
        expected = {model + '_rtc' for model in REMOVED}
        self.assertEqual({m['id'] for m in models['models']}, expected)
        self.assertTrue(all(m['rtc'] for m in models['models']))
        self.assertEqual(models['selected'], 'grasp_v3_step13000_rtc')
        config = json.loads((ROOT / 'vla_models.json').read_text())
        self.assertEqual(config['default'], 'grasp_v3_step13000_rtc')
        for model in REMOVED:
            self.error('UNKNOWN_MODEL', 400, self.client.vla_grasp, model=model)
            self.error('UNKNOWN_MODEL', 400, self.client.preflight, model=model)
        # RTC wrappers still depend on the installed synchronous implementation.
        installer = (ROOT / 'install_b70.sh').read_text()
        for model in config['models']:
            wrapper = Path(model['command'][0]).name
            self.assertTrue((ROOT / 'vla' / wrapper).exists())
            self.assertIn('vla/' + wrapper, installer)
            self.assertIn('vla/' + wrapper.replace('_rtc.sh', '.sh'), installer)
        self.assertIn('api_docs.html', installer)

    def test_curl_local_pages(self):
        # Exercises the exact manual smoke commands, using fake ROS and localhost only.
        for path, marker in (('/api-docs', '接口文档 / 一键测试'), ('/', 'href="/api-docs"')):
            result = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--max-time', '5',
                                     self.url + path], capture_output=True, text=True, check=True)
            self.assertIn(marker, result.stdout)
        self.assertEqual(self.commands('arm') + self.commands('vla') + self.commands('down'), [])


class DocumentationConsistencyTests(unittest.TestCase):
    def test_error_codes_match_api_documentation(self):
        html = (ROOT / 'api_docs.html').read_text()
        rows = json.loads(html.split('const errorCodes=', 1)[1].split(';\n', 1)[0])
        markdown = (ROOT / 'docs/API.md').read_text()
        table = markdown.split('| code | HTTP / 动作错误 | 含义 |', 1)[1].split('\n\n', 1)[0]
        expected = [[column.strip().replace('`', '') for column in line.strip('|').split('|')]
                    for line in table.splitlines() if '`' in line]
        self.assertEqual(rows, expected)

    def test_page_request_and_polling_logic(self):
        node = Path.home() / '.openagents/nodejs/bin/node'
        if not node.is_file():
            import shutil
            found = shutil.which('node')
            if not found:
                self.skipTest('Node.js unavailable for page logic test')
            node = Path(found)
        result = subprocess.run([str(node), str(ROOT / 'tests/api_docs_script_test.js')],
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
