"""Tests for the command-line interface"""

from __future__ import annotations

import yaml
import pytest

from param_ens_gen.cli import main


def test_main_no_subcommand_exits(mocker):
    """main() exits with an error when no subcommand is given."""
    mocker.patch("sys.argv", ["param_ens_gen"])
    with pytest.raises(SystemExit):
        main()


def test_main_run_missing_config_arg_exits(mocker):
    """main() exits when the config argument is missing."""
    mocker.patch("sys.argv", ["param_ens_gen", "run"])
    with pytest.raises(SystemExit):
        main()


def test_main_run_config_not_found_exits(mocker, tmp_path):
    """main() exits with code 1 when the config file does not exist."""
    mocker.patch("sys.argv", ["param_ens_gen", "run", str(tmp_path / "missing.yaml")])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_main_run_config_not_a_mapping_exits(mocker, tmp_path):
    """main() exits with code 1 when the YAML is not a mapping."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- item1\n- item2\n")

    mocker.patch("sys.argv", ["param_ens_gen", "run", str(config_file)])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_main_run_invalid_ensemble_type_exits(mocker, tmp_path):
    """main() exits with code 1 when ensemble_type is invalid."""
    config = {
        "ensemble_type": "NotReal",
        "param_dir": str(tmp_path),
        "ensemble_dir": str(tmp_path / "out"),
        "file_prefix": "test",
        "default_param_file": str(tmp_path / "default.nc"),
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))

    mocker.patch("sys.argv", ["param_ens_gen", "run", str(config_file)])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_main_run_calls_create_ensemble(
    mocker, tmp_path, ensemble_param_dir, default_param_file, posterior_config_file
):
    """main() calls create_ensemble() on a valid config."""
    config = {
        "ensemble_type": "LatinHypercube",
        "param_dir": str(ensemble_param_dir),
        "ensemble_dir": str(tmp_path / "out"),
        "file_prefix": "test",
        "default_param_file": str(default_param_file),
        "ensemble_members": 3,
        "posterior_sources": str(posterior_config_file),
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))

    mock_ensemble = mocker.MagicMock()
    mock_ensemble.num_params = 6
    mock_ensemble.ensemble_dir = tmp_path / "out"

    mocker.patch("sys.argv", ["param_ens_gen", "run", str(config_file)])
    mocker.patch(
        "param_ens_gen.cli.ParamEnsemble.from_dict", return_value=mock_ensemble
    )

    main()

    mock_ensemble.create_ensemble.assert_called_once()


def test_main_run_prints_summary(
    mocker,
    tmp_path,
    ensemble_param_dir,
    default_param_file,
    posterior_config_file,
    capsys,
):
    """main() prints ensemble type, param count, and completion message."""
    config = {
        "ensemble_type": "LatinHypercube",
        "param_dir": str(ensemble_param_dir),
        "ensemble_dir": str(tmp_path / "out"),
        "file_prefix": "test",
        "default_param_file": str(default_param_file),
        "ensemble_members": 3,
        "posterior_sources": str(posterior_config_file),
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))

    mock_ensemble = mocker.MagicMock()
    mock_ensemble.num_params = 6
    mock_ensemble.ensemble_dir = tmp_path / "out"

    mocker.patch("sys.argv", ["param_ens_gen", "run", str(config_file)])
    mocker.patch(
        "param_ens_gen.cli.ParamEnsemble.from_dict", return_value=mock_ensemble
    )

    main()

    captured = capsys.readouterr()
    assert "LatinHypercube" in captured.out
    assert "6" in captured.out
    assert "complete" in captured.out


def test_main_run_value_error_exits(mocker, tmp_path):
    """main() exits with code 1 when ParamEnsemble.from_dict raises ValueError."""
    config = {
        "ensemble_type": "LatinHypercube",
        "param_dir": str(tmp_path),
        "ensemble_dir": str(tmp_path / "out"),
        "file_prefix": "test",
        "default_param_file": str(tmp_path / "default.nc"),
        "ensemble_members": 3,
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))

    mocker.patch("sys.argv", ["param_ens_gen", "run", str(config_file)])
    mocker.patch(
        "param_ens_gen.cli.ParamEnsemble.from_dict",
        side_effect=ValueError("bad config"),
    )

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_main_run_file_not_found_exits(mocker, tmp_path):
    """main() exits with code 1 when ParamEnsemble.from_dict raises FileNotFoundError."""
    config = {
        "ensemble_type": "LatinHypercube",
        "param_dir": str(tmp_path),
        "ensemble_dir": str(tmp_path / "out"),
        "file_prefix": "test",
        "default_param_file": str(tmp_path / "default.nc"),
        "ensemble_members": 3,
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))

    mocker.patch("sys.argv", ["param_ens_gen", "run", str(config_file)])
    mocker.patch(
        "param_ens_gen.cli.ParamEnsemble.from_dict",
        side_effect=FileNotFoundError("missing file"),
    )

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
