"""Automates activities associated with building and deploying the python package."""

import ctypes
import subprocess
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
