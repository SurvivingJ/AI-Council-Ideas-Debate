import openai
from enum import Enum
import csv

class Side(Enum):
    FOR = 0
    AGAINST = 1

class Council:
    def __init__(self, topic):
        self.api_key = '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'
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

    def write_to_csv(self, content):
        with open('logs.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([content])

    def gen_ideas(self):
        if len(self.current_ideas) == 0:
            for id, member in self.members.items():
                idea = member['for_agent'].gen_iv(self.topic)
                print(idea)
                self.current_ideas.append(idea)
        print("+++++++++++++++++++")
        print("IDEAS:")
        print(self.current_ideas)
        print("+++++++++++++++++++")
        for idea in self.current_ideas:
            self.eval_idea(idea)

    def generate(self, prompt, temp):
        openai.api_key = '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'
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=temp,
            max_tokens=250
        )
        return response.choices[0].text

    def eval_idea(self, idea):
        judge = Judge()
        for id, member in self.members.items():
            for_arg = member['for_agent'].create_arg(self.topic, idea)
            opp_arg = member['opp_agent'].create_arg(self.topic, idea)
            judge.pass_judgement(self.topic, idea, for_arg, opp_arg)
            
class Judge(Council):
    def __init__(self):
        self.temp = 0.2
        self.scores = []

    def pass_judgement(self, topic, idea, for_arg, opp_arg):
        self.write_log(f"ARGS: forarg -> {for_arg} || opp_arg -> {opp_arg}")
        prompt = f"You are a judge. You receive arguments from two opposing views on a subject \
                  and weigh up the strength, logic and rationality of these arguments. You should \
                  work through each argument step by step, staying fully aware of any logical fallacies \
                  and look to provide unbiased judgement. Pass judgement on the following two arguments. The topic is {topic} and the idea is to use {idea} as an Instrumental Variable. \
                  The for argument is {for_arg}. The opposing argument is {opp_arg}. Which argument is \
                  more logically and rationally coherent? Respond as follows: Score each argument on a scale from 0 to 100 based on the criteria of logical cohesion, creativity, rationality. \
                    Repond with the following formatting, replacing the () with the relevant scores, ensuring to only respond with a number: For arg score = () ; Against arg score = ()"
        ##Respond 1 if for_arg, 2 if opp_arg.
        response = self.generate(prompt, self.temp)
        self.scores.append(response)
        print(response)
        print(f'SCORES: {self.scores}')
        self.write_log(response)
        self.write_to_csv(response)

class CouncilMember(Council):
    def __init__(self, profession, primer, side):
        self.temp = 0.8
        self.side = side
        self.profession = profession
        self.primer = primer

    def create_arg(self, topic, idea):
        if self.side == Side.FOR:
            keyword = "argue for"
        elif self.side == Side.AGAINST:
            keyword = "rebutt against"
        prompt = f"Given the topic {topic} and the idea of using {idea} as an Instrumental Variable, you operate on the {self.side} side of the debate. Generate a creative unique argument, \
                  using your expert knowledge as a {self.profession} in order to {keyword} the merit of using {idea} as an Instrumental Variable. Ensure it is a logical and rational argument."
        response = self.generate(prompt, self.temp)
        self.write_log(f"Arg: {self.side}")
        self.write_log(response)
        self.write_to_csv(f'Arg: {self.side}')
        self.write_to_csv(response)
                  
    def gen_iv(self, topic):
        with open("iv_primer.txt", 'r') as f:
            primer_info = f.read()
        prompt = f"{self.primer}. {primer_info}. Given your new knowledge on Instrumental Variables, generate 1 idea for an instrumental variables given the topic {topic}. It should be relevant to the independent \
                  variable, but not affect the dependent variable in the topic. Ensure to be creative, not give any ideas that you have given before, and draw on your expertise as {self.profession}. Respond simply with just the instrumental variable."
        iv = self.generate(prompt, self.temp)
        self.write_log("IV:")
        self.write_log(iv)
        self.write_to_csv('IV:')
        self.write_to_csv(iv)
        print(iv)
        return iv
        
if __name__ == "__main__":
    council = Council("Does an increase in years of secondary schooling cause an increase in wages? Independent Variable: Education | Dependent Variable: Wages")
    council.gen_ideas()
