from gpt4all import GPT4All
from enum import Enum

class Side(Enum):
    FOR = 0
    AGAINST = 1

class Council:
    def __init__(self, topic):
        self.members = {}
        self.topic = topic
        self.current_ideas = []
        with open("council.txt.", 'r') as f:
            for line in f:
                info = line.strip()
                csv_values = info.split(',')
                self.members[csv_values[0]] = {'profession': csv_values[1], 'primer': csv_values[2], 'for_agent': CouncilMember(csv_values[1], csv_values[2], Side.FOR), 'opp_agent': CouncilMember(csv_values[1], csv_values[2], Side.AGAINST)}

    def write_log(self, content):
        with open("logs.txt", "a") as f:
            f.write(content)
            f.write('\n')

    def gen_ideas(self):
        if len(self.current_ideas) == 0:
            for id, member in self.members.items():
                idea = member['for_agent'].gen_iv(self.topic)
                self.current_ideas.append(idea)
        for idea in self.current_ideas:
            self.eval_idea(idea)

    def eval_idea(self, idea):
        judge = Judge()
        for id, member in self.members.items():
            for_arg = member['for_agent'].create_arg(self.topic, idea)
            opp_arg = member['opp_agent'].create_arg(self.topic, idea)
            judge.pass_judgement(self.topic, idea, for_arg, opp_arg)
            
class Judge(Council):
    def __init__(self):
        self.model = GPT4All("ggml-model-gpt4all-falcon-q4_0.bin")
        self.session = self.model.chat_session()
        self.temp = 0.2

    def pass_judgement(self, topic, idea, for_arg, opp_arg):
        self.write_log(f"ARGS: forarg -> {for_arg} || opp_arg -> {opp_arg}")
        with self.session:
            primer = self.model.generate(prompt="You are a judge. You receive arguments from two opposing views on a subject \
                                         and weigh up the strength, logic and rationality of these arguments. You should \
                                         work through each argument step by step, staying fully aware of any logical fallacies \
                                         and look to provide unbiased judgement.", temp=self.temp)
            self.write_log(primer)
            response = self.model.generate(prompt=f"Pass judgement on the following two arguments. The topic is {topic} and the idea is to use {idea} as an Instrumental Variable. \
                                            The for argument is {for_arg}. The opposing argument is {opp_arg}. Which argument is \
                                            more logically and rationally coherent? Respond 1 if for_arg, 2 if opp_arg.", temp=self.temp)
            print(response)
            self.write_log(response)

class CouncilMember(Council):
    def __init__(self, profession, primer, side):
        self.model = GPT4All("ggml-model-gpt4all-falcon-q4_0.bin")
        self.session = self.model.chat_session()
        self.temp = 0.8
        self.side = side
        self.profession = profession
        self.primer = primer

    def create_arg(self, topic, idea):
        response = self.model.generate(prompt=f"Given the topic {topic} and the idea of using {idea} as an Instrumental Variable, you operate on the {self.side} side of the debate. Generate a creative unique argument, \
                                    using your expert knowledge as a {self.profession}. Ensure it is a logical and rational argument.", temp=self.temp)
        self.write_log(f"Arg: {self.side}")
        self.write_log(response)
                  
    def gen_iv(self, topic):
        with open("iv_primer.txt", 'r') as f:
            primer_info = f.read()
        with self.session:
            primer_resp = self.model.generate(prompt=self.primer, temp=self.temp)
            self.write_log(primer_resp)
            iv_prime_resp = self.model.generate(prompt=primer_info)
            iv = self.model.generate(prompt=f"Given your new knowledge on Instrumental Variables, generate 1 idea for an instrumental variables given the topic {topic}. It should be relevant to the independent \
                                variable, but not affect the dependent variable in the topic. Ensure to be creative, not give any ideas that you have given before, and draw on your expertise as {self.profession}. Respond simply with just the instrumental variable.")
            self.write_log("IV:")
            self.write_log(iv)
        print(iv)
        return iv
        
if __name__ == "__main__":
    GPT4All('ggml-model-gpt4all-falcon-q4_0.bin')
    council = Council("Does an increase in years of secondary schooling cause an increase in wages? Independent Variable: Education | Dependent Variable: Wages")
    council.gen_ideas()