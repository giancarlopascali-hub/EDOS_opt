import pathlib
import subprocess
import unittest


class ImportanceRebalancingTests(unittest.TestCase):
    def test_included_importances_sum_to_exactly_100(self):
        utils_uri = (pathlib.Path(__file__).parents[1] / 'static/js/utils.js').resolve().as_uri()
        script = f"""
class HTMLElement {{ constructor(col, value) {{ this.dataset={{col}}; this.value=value; }} }}
globalThis.HTMLElement = HTMLElement;
const inputs = [new HTMLElement('A','60'), new HTMLElement('B','20'), new HTMLElement('C','20')];
globalThis.window = {{ objectiveConfigs: {{A:{{included:true}},B:{{included:true}},C:{{included:true}}}} }};
globalThis.document = {{ querySelectorAll: () => inputs }};
const mod = await import('{utils_uri}');
mod.rebalanceRemaining(inputs[0], '33.3', '.importance');
const total = inputs.reduce((sum, item) => sum + Math.round(Number(item.value) * 10), 0);
if (total !== 1000) throw new Error(`Expected 1000 tenths, received ${{total}}`);
"""
        completed = subprocess.run(
            ['node', '--input-type=module', '--eval', script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
