from .base_coder import Coder
from .refine_prompts import RefinePrompts


class RefineCoder(Coder):
    """Refine something"""

    edit_format = "refine"
    gpt_prompts = RefinePrompts()
    repo_map = None
    suggest_shell_commands = False