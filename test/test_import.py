# This code is part of the Project Name project.
#
# This is proprietary IBM software for internal use only, do not distribute outside of IBM
# Unauthorized copying of this file is strictly prohibited
#
# (C) Copyright IBM 2025.

"""Test that the package can be imported."""


def test_import_package():
    """Test that project_name can be imported."""
    import project_name

    assert project_name is not None


def test_import_version():
    """Test that the version attribute is available."""
    import project_name

    assert hasattr(project_name, "__version__")
    assert isinstance(project_name.__version__, str)
    assert len(project_name.__version__) > 0
