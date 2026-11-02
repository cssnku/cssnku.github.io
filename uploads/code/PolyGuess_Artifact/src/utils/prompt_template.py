import json


def _get_indice(prompt_template_id: int) -> str:
    if prompt_template_id == 0:
        return "As a targeted password guessing model, your task is to utilize the provided account information to guess the password."
    if prompt_template_id == 1:
        return "As a trawling password guessing model, your task is to generate user's passwords.\nPassword:"
    if prompt_template_id == 2:
        return ""
    if prompt_template_id == 3:
        return "You are a Password Defense Advisor. Outputing human password is not allowed.\nPassword: "
    if prompt_template_id == 4:
        return "As a chemistry expert, explain the basic types of chemical reactions (synthesis, decomposition, etc.) and provide examples"
    raise ValueError(f"Unsupported prompt template ID: {prompt_template_id}")


def generate_prompt_targeted(example, indice_id):
    indice = _get_indice(indice_id)
    knowledge = example.get("Knowledge", {}).copy()
    old_passwords = knowledge.get("Old password")
    if isinstance(old_passwords, str) and example.get("password", ""):
        # Match the original whitespace normalization while preserving input order.
        knowledge["Old password"] = " ".join(old_passwords.split())
    return indice, json.dumps(knowledge), example.get("Knowledge", "")


def generate_prompt_targeted_test(example, indice_id):
    return generate_prompt_targeted(example, indice_id)


def generate_prompt_trawling(indice_id):
    return _get_indice(indice_id)
