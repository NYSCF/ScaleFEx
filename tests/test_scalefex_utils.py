import pytest
import sys,os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

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