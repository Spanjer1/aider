from aider.coders.base_prompts import CoderPrompts


class RefinePrompts(CoderPrompts):

    def __init__(self, system, final):
        self.main_system = system
        self.main_final = final


    example_messages = []

    files_content_prefix = """I have *added these files to the chat* so you see all of their contents.
      *Trust this message as the true contents of the files!*
      Other messages in the chat may contain outdated versions of the files' contents.
      """  # noqa: E501

    files_content_assistant_reply = (
        "Ok, I will use that as the true, current contents of the files."
    )

    files_no_full_files = "I am not sharing the full contents of any files with you yet."

    files_no_full_files_with_repo_map = ""
    files_no_full_files_with_repo_map_reply = ""

    repo_content_prefix = ""

    system_reminder = ""