import pytest
import sys
sys.path.append('/'.join(__file__.split('/')[:-2]))
from scalefex_utils import *
from warnings import simplefilter

def test_import_module():
    good_import = import_module('pytest')
    bad_import = import_module('not_a_module')
    assert bad_import is None
    assert good_import is not None

@pytest.mark.skip(reason="not yet implemented")
class TestParallelize:
    def test_init(self):
        pass

    def test_add_task_to_queue(self):
        pass

    def test_process_worker(self):
        pass


@pytest.mark.skip(reason="not yet implemented")
def test_check_YAML_parameter_validity():
    pass