from enum import Enum
import csv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
import json
import os
from gpt4all import GPT4All

class Side(Enum):
    FOR = 0
    AGAINST = 1

class Council:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_AI_COUNCIL_KEY")
        if self.api_key is None:
            raise ValueError("Secret key not set in environment variables.")
        self.current_ideas = []
        
        # Topic
        with open("topic.txt", 'r') as f:
            self.topic = f.read()
        
        # Sentiment Analysis
        nltk.download('vader_lexicon')
        
        # Spawn members
        self.members = {}
        with open("council.txt.", 'r') as f:
            for line in f:
                info = line.strip()
                csv_values = info.split(',')
                self.members[csv_values[0]] = {'profession': csv_values[1], 'primer': csv_values[2], 'for_agent': CouncilMember(csv_values[1], csv_values[2], Side.FOR), 'opp_agent': CouncilMember(csv_values[1], csv_values[2], Side.AGAINST)}
        
        # Spawn judges
        self.judges = []
        for _ in range(1):#####
            judge = Judge()
            self.judges.append(judge)

    def write_log(self, content, filename='logs.txt'):
        ''' Write into a text file '''
        with open(filename, "a") as f:
            f.write(str(content))
            f.write('\n')

    def write_to_csv(self, content, filename='logs.csv'):
        ''' Write into a csv file '''
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([content])

    def gen_ideas(self):
        ''' Generate ideas from each member of the council '''
        if len(self.current_ideas) == 0:
            for i in range(1):
                for id, member in self.members.items():
                    for i in range(1):
                        idea = member['for_agent'].idea_gen(self.topic)
                        print(idea)
                        self.current_ideas.append(idea)
        print("+++++++++++++++++++")
        print("IDEAS:")
        print(self.current_ideas)
        print("+++++++++++++++++++")
        idea_results = {}
        for i in range(2):
            idea = self.current_ideas[random.randint(0, len(self.current_ideas)-1)]
            total_for_score, total_opp_score = self.eval_idea(idea)
            idea_results[idea] = [total_for_score, total_opp_score]
        print(idea_results)

        self.write_log(idea_results, 'results.txt')
        best_idea = ''
        best_score = 0
        for idea, score in idea_results.items():
            total_score = score[0] + score[1]
            if total_score > best_score:
                best_idea = idea
                best_score = total_score
        print(best_score)
        print(best_idea)
        best_idea_str = json.dumps(best_idea)
        self.write_log(best_score, 'best_option.txt')
        self.write_log(best_idea_str, 'best_option.txt')



    def generate(self, prompt, temp):
        ''' Model call to OpenAI '''
        self.model = GPT4All("ggml-model-gpt4all-falcon-q4_0.bin")
        self.session = self.model.chat_session()
        with self.session:
            response = self.model.generate(prompt=prompt, temp=temp, max_tokens=500)
        return response

    def eval_idea(self, idea):
        ''' 
        Evaluates ideas
        1. Each member creates an argument for and against the idea
        2. Each judge passes judgement on the idea
        '''
        total_for_score = 0
        total_opp_score = 0
        for id, member in self.members.items():
            for_arg = member['for_agent'].create_arg(self.topic, idea)
            opp_arg = member['opp_agent'].create_arg(self.topic, idea)
            for_rebut = member['for_agent'].rebut_arg(self.topic, idea, opp_arg)
            opp_rebut = member['opp_agent'].rebut_arg(self.topic, idea, for_arg)
            for judge in self.judges:
                for_score, opp_score, for_response, opp_response = judge.pass_judgement(self.topic, idea, for_arg, opp_arg)
                for_score_rebut, opp_score_rebut, for_response_rebut, opp_response_rebut = judge.pass_judgement(self.topic, idea, for_rebut, opp_rebut)
                print(f"FOR SCORE: {for_score}| FOR REBUT: {for_rebut}| OPP SCORE: {opp_score}| OPP REBUT: {opp_rebut}")
                total_for_score += for_score
                total_for_score += for_score_rebut
                total_opp_score += opp_score
                total_opp_score += opp_score_rebut
        print(f"Total Opposition: {total_opp_score}")
        print(f"Total For: {total_for_score}")
        self.write_log("++++++++++++++++++++++++", "scoring.txt")
        self.write_log(f"For Arg: {for_arg}", "scoring.txt")
        self.write_log(f"For Judgement: {for_response}", "scoring.txt")
        self.write_log(f"For Rebuttal Judgement: {for_response_rebut}", "scoring.txt")
        self.write_log(f"Total For: {total_for_score}", "scoring.txt")
        self.write_log(f"Opp Arg: {opp_arg}", "scoring.txt")
        self.write_log(f"Opposition Judgement: {opp_response}", "scoring.txt")
        self.write_log(f"Opposition Rebuttal Judgement: {opp_response_rebut}", "scoring.txt")
        self.write_log(f"Total Opposition: {total_opp_score}", "scoring.txt")
        self.write_log("++++++++++++++++++++++++", "scoring.txt")
        return total_for_score, total_opp_score
    
    def sentiment_analysis(self, judgement):
        ''' Sentiment Analysis of the Judgement from the judges '''
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(judgement)
        if sentiment_scores['compound'] >= 0.05:
            sentiment = "Positive"
            score = 1
        elif sentiment_scores['compound'] <= -0.05:
            sentiment = "Negative"
            score = -1
        else:
            sentiment = "Neutral"
            score = 0

        print(f"Sentiment: {sentiment}")
        self.write_log(sentiment)
        return score

class Judge(Council):
    def __init__(self):
        self.temp = 0.3
        self.scores = []
        self.intended_use = 'Trading Strategy'

    def pass_judgement(self, topic, idea, for_arg, opp_arg):
        ''' Pass judgement on the quality of the idea '''
        self.write_log(f"ARGS: forarg -> {for_arg} || opp_arg -> {opp_arg}")
        for_prompt = f"You are a judge. You receive arguments from two opposing views on a subject \
                  and weigh up the strength, logic and rationality of these arguments. You should \
                  work through each argument step by step, staying fully aware of any logical fallacies \
                  and look to provide unbiased judgement. Pass judgement on the following argument. The topic is {topic} and the idea is to use {idea} as {self.intended_use}. \
                  The for argument is {for_arg}. Respond with judgement on the quality of this argument. Ensure you are impartial, logical, rational. Use positive language if you think the \
                    argument was good, and negative language if you think it was poor. Be ultra-critical."
        opp_prompt = f"You are a judge. You receive arguments from two opposing views on a subject \
                  and weigh up the strength, logic and rationality of these arguments. You should \
                  work through each argument step by step, staying fully aware of any logical fallacies \
                  and look to provide unbiased judgement. Pass judgement on the following argument. The topic is {topic} and the idea is to use {idea} as {self.intended_use}. \
                  The for argument is {opp_arg}. Respond with judgement on the quality of this argument. Ensure you are impartial, logical, rational. Use positive language if you think the \
                    argument was good, and negative language if you think it was poor. Be ultra-critical."

        for_response = self.generate(f"{for_prompt}", self.temp)
        opp_response = self.generate(f"{opp_prompt}", self.temp)

        self.write_log(for_response)
        self.write_to_csv(for_response)
        self.write_log(opp_response)
        self.write_to_csv(opp_response)
        for_score = self.sentiment_analysis(for_response)
        opp_score = self.sentiment_analysis(opp_response)
        if for_score > opp_score:
            self.write_log('+++++++++++++++++++++++', 'sentiment.txt')
            self.write_log("FOR ARG", 'sentiment.txt')
            self.write_log(for_arg, 'sentiment.txt')
            self.write_log(idea, 'sentiment.txt')
            self.write_log(for_response, 'sentiment.txt')
            self.write_log('+++++++++++++++++++++++', 'sentiment.txt')
        elif opp_score > for_score:
            self.write_log('+++++++++++++++++++++++', 'sentiment.txt')
            self.write_log("OPP ARG", 'sentiment.txt')
            self.write_log(opp_arg, 'sentiment.txt')
            self.write_log(idea, 'sentiment.txt')
            self.write_log(opp_response, 'sentiment.txt')
            self.write_log('+++++++++++++++++++++++', 'sentiment.txt')
    
        return for_score, opp_score, for_response, opp_response

class CouncilMember(Council):
    def __init__(self, profession, primer, side):
        self.temp = 0.9
        self.side = side
        self.profession = profession
        self.primer = primer
        self.intended_use = 'Simple Parsimonious Trading Strategy'

    def create_arg(self, topic, idea):
        ''' Create arguments for the idea '''
        if self.side == Side.FOR:
            keyword = "argue for"
        elif self.side == Side.AGAINST:
            keyword = "rebutt against"

        prompt = f"Given the topic {topic} and the idea of using {idea} as a {self.intended_use}, you operate on the {self.side} side of the debate. Generate a creative unique argument, \
                  using your expert knowledge as a {self.profession} in order to {keyword} the merit of using {idea} as {self.intended_use}. Ensure it is a logical and rational argument."
        response = self.generate(prompt, self.temp)
        self.write_log(f"Arg: {self.side}")
        self.write_log(response)
        self.write_to_csv(f'Arg: {self.side}')
        self.write_to_csv(response)
        return response
    
    def rebut_arg(self, topic, idea, arg):
        ''' Rebut argument '''
        prompt = f"Given the topic {topic} and the idea of using {idea} as a {self.intended_use}, you operate on the {self.side} side of the debate. Generate a creative unique rebuttal \
                  using your expert knowledge as a {self.profession} in order to rebut the following argument: {arg}"
        response = self.generate(prompt, self.temp)
        self.write_log(f"Rebuttal: {self.side}")
        self.write_log(response)
        return response

    def idea_gen(self, topic):
        ''' Generate ideas '''
        prompt = f"Given the topic: {topic}; generate an interesting, unique idea for a {self.intended_use}"
        idea = self.generate(prompt, self.temp)
        self.write_log("Idea:")
        self.write_log(idea)
        self.write_to_csv('Idea:')
        self.write_to_csv(idea)
        print(idea)
        return idea
        
if __name__ == "__main__":
    # Create the council and begin the idea generating process
    GPT4All('ggml-model-gpt4all-falcon-q4_0.bin')
    council = Council()
    council.gen_ideas()