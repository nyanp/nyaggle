import os
import tempfile
import shutil
import uuid
import pytest


@pytest.fixture(scope='function', autouse=True)
def tmpdir_name():
    path = None
    try:
        path = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex)
        yield path
    finally:
        if path:
            shutil.rmtree(path, ignore_errors=True)
