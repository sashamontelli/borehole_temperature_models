"""Automates activities associated with building and deploying the python package."""

import ctypes
import subprocess
import re
import sys
import textwrap

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import typer

from typer.core import TyperGroup


# ----------------------------------------------------------------------
class NaturalOrderGrouper(TyperGroup):
    # pylint: disable=missing-class-docstring
    # ----------------------------------------------------------------------
    def list_commands(self, *args, **kwargs):  # pylint: disable=unused-argument
        return self.commands.keys()


# ----------------------------------------------------------------------
app                                         = typer.Typer(
    cls=NaturalOrderGrouper,
    help=__doc__,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_enable=False,
)


# ----------------------------------------------------------------------
@app.command("Build", no_args_is_help=False)
def Build(
    additional_args: list[str]=typer.Option([], "--arg", help="Additional arguments passed to the build command."),
    verbose: bool=typer.Option(False, "--verbose", help="Write verbose information to the terminal."),
) -> None:
    """Builds the python package."""

    sys.stdout.write("Building...")
    sys.stdout.flush()

    result = _ExecuteCommand(
        "python -m build{}".format(
            "" if not additional_args else " {}".format(" ".join('"{}"'.format(arg) for arg in additional_args)),
        ),
    )

    sys.stdout.write("DONE ({})!\n\n".format(result.returncode))

    result.RaiseOnError()

    if verbose:
        sys.stdout.write(result.output)


# ----------------------------------------------------------------------
@app.command("Publish", no_args_is_help=True)
def Publish() -> None:
    """Publishes the python package to PyPi."""

    raise Exception("TODO: Not implemented yet.")


# ----------------------------------------------------------------------
@app.command("UpdateVersion", no_args_is_help=False)
def UpdateVersion(
    verbose: bool=typer.Option(False, "--verbose", help="Write verbose information to the terminal."),
) -> None:
    """Updates the version of the package based on GitHub changes."""

    this_dir = Path(__file__).parent

    # Calculate the version
    sys.stdout.write("Calculating version...")
    sys.stdout.flush()

    command_line = 'docker run --rm -v "{}:/local" dbrownell/autosemver:0.6.0 --path /local --no-metadata --quiet'.format(this_dir)

    result = _ExecuteCommand(command_line)

    sys.stdout.write("DONE ({})!\n\n".format(result.returncode))

    result.RaiseOnError()

    version = result.output.strip()

    # Update the source
    sys.stdout.write("Updating source...")
    sys.stdout.flush()

    init_filename = this_dir / "src" / "borehole_temperature_models" / "__init__.py"
    assert init_filename.is_file(), init_filename

    with init_filename.open(encoding="utf-8") as f:
        content = f.read()

    new_content = re.sub(
        r'^__version__ = ".+?"$',
        f'__version__ = "{version}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )

    with init_filename.open("w", encoding="utf-8") as f:
        f.write(new_content)

    sys.stdout.write("DONE!\n\n")


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class _ExecuteCommandResult(object):
    """Result returned by _ExecuteCommand."""

    # ----------------------------------------------------------------------
    returncode: int
    output: str

    error_command_line: Optional[str]       = field(default=None)

    # ----------------------------------------------------------------------
    def RaiseOnError(self) -> None:
        if self.returncode == 0:
            return

        assert self.error_command_line is not None

        raise Exception(
            textwrap.dedent(
                """\
                Command Line
                ------------
                {}

                Return Code
                -----------

                Output
                ------
                {}
                """,
            ).format(
                self.error_command_line.rstrip(),
                self.returncode,
                self.output.rstrip(),
            ),
        )


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def _ExecuteCommand(
    command_line: str,
    cwd: Optional[Path]=None,
) -> _ExecuteCommandResult:
    result = subprocess.run(
        command_line,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
    )

    content = result.stdout.decode("utf-8")

    returncode = result.returncode
    if returncode <= 255:
        returncode = ctypes.c_byte(returncode).value
    else:
        returncode = ctypes.c_long(returncode).value

    return _ExecuteCommandResult(
        returncode,
        content,
        command_line if returncode != 0 else None,
    )


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    app()
