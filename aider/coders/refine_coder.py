from .base_coder import Coder
from .refine_prompts import RefinePrompts


class RefineCoder(Coder):
    """Refine something"""

    edit_format = "refine"
    repo_map = None
    use_repo_map = False
    suggest_shell_commands = False