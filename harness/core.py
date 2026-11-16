from dataclasses import dataclass, field
import hashlib
import json

@dataclass
class Harness:
    system_prompt: str
    parent: str | None = None
    tag: str = "llm"
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        key = json.dumps({"sp": self.system_prompt, "x": self.extra}, sort_keys=True)
        return hashlib.md5(key.encode()).hexdigest()[:10]

    def message(self, question: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]
