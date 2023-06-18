"""Base class for SWAN components.

How to subclass
---------------

- Define a new `model_type` Literal for the subclass (set the default value to avoid
  having to set it when instantiating the object)
- Overwrite the cmd method to return the SWAN input file string

"""
import logging
from typing import Literal, Optional
from abc import abstractmethod
from pydantic import Field

from rompy.core import RompyBaseModel


logger = logging.getLogger(__name__)

MAX_LENGTH = 132
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
    list
        Split string.

    """
    if len(cmd) <= max_length:
        return [cmd]

    split_index = cmd.rfind(" ", 0, max_length - 1 - spaces)

    if split_index == -1:
        split_index = max_length

    return [cmd[:split_index]] + split_string(cmd[split_index + 1 :])


class BaseComponent(RompyBaseModel):
    """Base class for SWAN components.

    Parameters
    ----------
    model_type: Literal["component"]
        Model type discriminator.

    Behaviour
    ---------
    - Define a render method to render the component to a cmd string.
    - Restrict arguments to the defined ones.

    """

    model_type: Literal["component"] = Field(description="Model type discriminator")

    class Config:
        """Configure the model."""

        extra = "forbid"

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
