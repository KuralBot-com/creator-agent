import os


class PromptLoader:
    """Reads system and user prompt files from a directory at startup."""

    def __init__(self, prompt_dir: str):
        system_path = os.path.join(prompt_dir, "system.txt")
        user_path = os.path.join(prompt_dir, "user.txt")

        if not os.path.isfile(system_path):
            raise RuntimeError(f"System prompt not found: {system_path}")
        if not os.path.isfile(user_path):
            raise RuntimeError(f"User prompt template not found: {user_path}")

        with open(system_path, encoding="utf-8") as f:
            self._system = f.read().strip()
        with open(user_path, encoding="utf-8") as f:
            self._user_template = f.read().strip()

        if not self._system:
            raise RuntimeError(f"System prompt is empty: {system_path}")
        if not self._user_template:
            raise RuntimeError(f"User prompt template is empty: {user_path}")

    def get_system_prompt(self) -> str:
        return self._system

    def get_user_prompt(self, prompt: str) -> str:
        return self._user_template.format(prompt=prompt)
