import json
from dataclasses import dataclass, field


@dataclass
class Guess:
    model_input: str
    guess_list: list[str]
    pro_list: list[int]
    target: str
    min_cracked_num: int = -1


@dataclass
class Log_config:
    log_f: str
    log_guess_number: list[int] = field(default_factory=lambda: [1, 10, 50, 100, 200, 1000])


@dataclass
class Desc:
    out_f: str
    counter: int
    guesses: list[Guess]


@dataclass
class Rdesc:
    path: str
    desc: str


@dataclass
class Mresult:
    Rdescs: list[Rdesc]
    below_success_rate: list[list[int]]
    at_success_rate: list[list[int]]
    min_auto: list


class pm_guesser:

    def __init__(self, f, log_config: Log_config, save_interval=10) -> None:
        self.state_dic: Desc = Desc(f, 0, [])
        self.save_interval = save_interval
        self.log_config = log_config
        self.log_suc_number = [0 for _ in range(len(self.log_config.log_guess_number))]

    def update(self, input, target, guess_list):
        pwds = [item[0] for item in guess_list]
        pros = [item[1] for item in guess_list]
        min_guess = 0

        if target in pwds:
            min_guess = pwds.index(target) + 1

        guess = Guess(input, pwds, pros, target, min_guess)
        self.state_dic.counter += 1
        self.state_dic.guesses.append(guess)

        if self.state_dic.counter % self.save_interval == 0:
            self._save(self.state_dic.counter - self.save_interval, self.state_dic.counter)

        return min_guess

    def _save(self, begin, end):
        with open(self.state_dic.out_f, "a") as f, open(self.log_config.log_f, "a") as l:
            for i in range(begin, end):
                save_template = {
                    "index": i,
                    "min_cracked_guess_number": 0,
                    "model_input": "",
                    "real password": "",
                    "outputPasswords": []
                }

                target = self.state_dic.guesses[i].target
                guesses = self.state_dic.guesses[i].guess_list

                save_template["min_cracked_guess_number"] = self.state_dic.guesses[i].min_cracked_num
                save_template["model_input"] = self.state_dic.guesses[i].model_input
                save_template["real password"] = target
                save_template["outputPasswords"] = list(zip(guesses, self.state_dic.guesses[i].pro_list))

                json.dump(save_template, f)
                f.write("\n")

                for index, guess_number in enumerate(self.log_config.log_guess_number):
                    if (
                        self.state_dic.guesses[i].min_cracked_num != 0 and
                        self.state_dic.guesses[i].min_cracked_num <= guess_number
                    ):
                        self.log_suc_number[index] += 1

                l.write(
                    f"id:{i} max_guess_size:{len(self.state_dic.guesses[i].guess_list)}  "
                    f"guess number:{self.log_config.log_guess_number}  "
                    f"cracked number:{self.log_suc_number} "
                    f"cracked rate:{[round(m / (i + 1), 4) for m in self.log_suc_number]}\n"
                )

                f.flush()
                l.flush()
