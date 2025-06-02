"""
SWAN Components Base Module

This module provides the base classes for SWAN components in the ROMPY framework.

How to subclass:
1. Define a new `model_type` Literal for the subclass
2. Overwrite the `cmd` method to return the SWAN input file string
"""

# Standard library imports
from typing import Literal, Optional
from abc import abstractmethod

# Third-party imports
from pydantic import ConfigDict, Field

# Local imports
from rompy.core.types import RompyBaseModel
from rompy.core.logging import get_logger

# Initialize the logger
logger = get_logger(__name__)

# Constants
MAX_LENGTH = 180
SPACES = 4


def split_string(cmd: str, max_length: int = MAX_LENGTH, spaces: int = 4) -> list:
    """Split command cmd if longer than max_length.

    Parameters
    ----------
    cmd: str
        SWAN CMD string to split.
    max_length: int
        Maximum length of each split string.
    spaces: int
        Number of spaces in each identation before each new line.

    Returns
    -------
    split_cmd: list
        List of split CMD strings.

    """
    if len(cmd) <= max_length:
        return [cmd]

    split_index = cmd.rfind(" ", 0, max_length - 1 - spaces)

    if split_index == -1:
        split_index = max_length

    return [cmd[:split_index]] + split_string(cmd[split_index + 1 :])


class BaseComponent(RompyBaseModel):
    """Base class for SWAN components.

    This class is not intended to be used directly, but to be subclassed by other
    SWAN components to implement the following common behaviour:

    * Define a `render()` method to render a CMD string from the component
    * Split the rendered CMD if longer than 132 characters
    * Forbid extra arguments so only implemented fields must be specified

    """

    model_type: Literal["component"] = Field(description="Model type discriminator")
    model_config = ConfigDict(extra="forbid")

    def _render_split_cmd(self, cmd_line: str) -> str:
        """Split cmd_line if longer than MAX_LENGTH.

        Longer strings are recursively split by inserting a SWAN line continuation
        character `&` follwed by newline and identation until no line it soo long.

        Parameters
        ----------
        cmd_line: str
            Command line to split.

        Returns
        -------
        str
            Split command line.

        """
        # Split cmd at existing newlines
        cmd_lines = cmd_line.split("\n")
        # Split each line before max_length
        cmds = []
        for cmd in cmd_lines:
            cmds.extend(split_string(cmd, max_length=MAX_LENGTH, spaces=SPACES))
        # Joining lines
        return f" &\n{SPACES * ' '}".join(cmds)

    @abstractmethod
    def cmd(self) -> str | list:
        """Return the string or list of strings to render the component to the CMD."""
        pass

    def render(self, cmd: Optional[str | list] = None) -> str:
        """Render the component to a string.

        Parameters
        ----------
        cmd: Optional[str | list]
            Command string or list of command strings to render, by default self.cmd().

        Returns
        -------
        cdmstr: str
            The rendered command file component.

        """
        cmd_lines = cmd or self.cmd()
        if isinstance(cmd_lines, str):
            cmd_lines = [cmd_lines]
        cmd_lines = [self._render_split_cmd(cmd_line) for cmd_line in cmd_lines]
        return "\n".join(cmd_lines)


class MultiComponents(BaseComponent):
    """Mixin for components that can have multiple components."""

    components: list = Field(description="The components to render")

    def cmd(self) -> list[str]:
        repr = []
        for component in self.components:
            repr += [component.cmd()]
        return repr
