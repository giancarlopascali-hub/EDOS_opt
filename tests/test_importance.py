import pathlib
import shutil
import subprocess
import unittest

NODE_AVAILABLE = shutil.which('node') is not None


@unittest.skipUnless(NODE_AVAILABLE, 'Node.js is required for JavaScript utility tests')
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

    def test_categorical_choice_csv_round_trip_preserves_embedded_commas(self):
        utils_uri = (pathlib.Path(__file__).parents[1] / 'static/js/utils.js').resolve().as_uri()
        script = f"""
globalThis.window = {{}};
const mod = await import('{utils_uri}');
const formatted = mod.formatCategoricalChoices(['1,4-Dioxane', 'THF']);
if (formatted !== '"1,4-Dioxane", THF') throw new Error(`Unexpected format: ${{formatted}}`);
const parsed = mod.parseCategoricalChoices(formatted);
if (JSON.stringify(parsed) !== JSON.stringify(['1,4-Dioxane', 'THF'])) {{
    throw new Error(`Unexpected parse result: ${{JSON.stringify(parsed)}}`);
}}
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
