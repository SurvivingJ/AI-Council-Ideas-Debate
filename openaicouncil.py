<<<<<<< HEAD
from openai import OpenAI
=======
'''
Title: AI Council
Author: James McBurnie
Github Link: https://github.com/SurvivingJ/AI-Council-Ideas-Debate
'''
import openai
>>>>>>> 79df2b3259772d79ef5eca94628112422b8b569c
from enum import Enum
import csv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
import json
import os
import time

class Side(Enum):
    FOR = 0
    AGAINST = 1

class Council:
    def __init__(self, topic):
        self.api_key = os.environ.get("OPENAI_AI_COUNCIL_KEY")
        if self.api_key is None:
            raise ValueError("Secret key not set in environment variables.")
        self.current_ideas = []

        self.client = OpenAI(api_key=self.api_key)
        # Topic
        self.topic = topic
        
        # Sentiment Analysis
        nltk.download('vader_lexicon')
        
        # Spawn members
        self.members = {}
        with open("council.txt.", 'r') as f:
            for line in f:
                info = line.strip()
                csv_values = info.split(',')
                assis_id = csv_values[2]
                for_thread = self.client.beta.threads.create()
                opp_thread = self.client.beta.threads.create()
                self.members[csv_values[0]] = {'assis_id': assis_id, 'for_agent': CouncilMember(assis_id, for_thread, Side.FOR), 'opp_agent': CouncilMember(assis_id, opp_thread, Side.AGAINST)}
        
        # Spawn judges
        self.judges = []
        for _ in range(1):#####
            thread_id = self.client.beta.threads.create()
            judge = Judge('asst_AykIYwVycEedcLYLxYccVPvo', thread_id)
            self.judges.append(judge)

    def write_log(self, content, filename='logs.txt'):
        ''' Write into a text file '''
        with open(filename, "a", encoding='utf-8') as f:
            f.write(str(content))
            f.write('\n')

    def write_to_csv(self, content, filename='logs.csv'):
        ''' Write into a csv file '''
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([content])

    def gen_ideas(self):
        ''' Generate ideas from each member of the council '''
        if len(self.current_ideas) == 0:
            for id, member in self.members.items():
                for i in range(1):
                    idea = member['for_agent'].idea_gen(self.topic)
                    print(idea)
                    self.current_ideas.append(idea)
        print("+++++++++++++++++++")
        print("IDEAS:")
        print(self.current_ideas)
        self.write_log(str(self.current_ideas), 'ideas.txt')
        print("+++++++++++++++++++")
        idea_results = {}
        ideas_used = []
        for i in range(3):
            _ = True
            while _:
                index = random.randint(0, len(self.current_ideas)-1)
                if index not in ideas_used:
                    _ = False
                
            idea = self.current_ideas[index]
            #idea = self.current_ideas[i]
            ideas_used.append(index)
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

    def wait_on_run(self, run, thread):
        while run.status == "queued" or run.status == "in_progress":
            run = council.client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            time.sleep(0.5)
        return run

    def submit_message(self, assistant_id, thread, user_message):
        council.client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=user_message
        )
        return council.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )
    
    def get_response(self, thread):
        return council.client.beta.threads.messages.list(thread_id=thread.id, order="asc")
    
    def parse_response(self, response):
        messages = []
        for thread_message in response.data:
            for content in thread_message.content:
                    message_text = content.text.value
                    messages.append(message_text)
        return messages
    def generate(self, prompt, assis_id, thread):
        ''' Model call to OpenAI '''
        run = self.submit_message(assis_id, thread, prompt)
        run_result = self.wait_on_run(run, thread)

        response = self.get_response(thread)
        messages = self.parse_response(response)
        return messages[-1]

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
    def __init__(self, assis_id, thread_id):
        self.scores = []
        self.assis_id = assis_id
        self.thread_id = thread_id

    def pass_judgement(self, topic, idea, for_arg, opp_arg):
        ''' Pass judgement on the quality of the idea '''
        self.write_log(f"ARGS: forarg -> {for_arg} || opp_arg -> {opp_arg}")
        for_prompt = f"Pass judgement on the following argument. The topic is {topic} and the idea is {idea}. \
                  The for argument is {for_arg}."
        opp_prompt = f"Pass judgement on the following argument. The topic is {topic} and the idea is {idea}. \
                  The opp argument is {opp_arg}."

        for_response = self.generate(for_prompt, self.assis_id, self.thread_id)
        opp_response = self.generate(opp_prompt, self.assis_id, self.thread_id)

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
    def __init__(self, assis_id, thread_id, side):
        self.side = side
<<<<<<< HEAD
        self.assis_id = assis_id
        self.thread_id = thread_id
=======
        self.profession = profession
        self.primer = primer
        self.intended_use = 'A low to medium frequency algorithmic trading strategy. We will be trading stocks on the Nasdaq. Try to use at least 2-3 different types and sources of data. Here is the data we have available in the form of a dictionary: \
        {US Equity Security Master: Corporate action data source for splits; dividends; mergers; acquisitions; IPOs; and delistings, \
        Bitcoin Metadata: Bitcoin processing fundamental data such as hash rate; miner revenue and number of transactions, \
        Data Link dataset by Nasdaq: Data on Nasdaq companies, \
        Treasury: US Daily Treasury Yield Rates, \
        US Energy Info: Supply and demand information for US Crude Products, \
        US Federal Reserve: FRED Economic Datasets, \
        US Fundamental Data: Corporate Fundamental Data for fine universe selection based on industry classification and underlying company performance indicators, \
        US Futures Security Master: Rolling reference data for popular CME Futures contracts}'
>>>>>>> 79df2b3259772d79ef5eca94628112422b8b569c

    def create_arg(self, topic, idea):
        ''' Create arguments for the idea '''
        if self.side == Side.FOR:
            keyword = "argue for"
        elif self.side == Side.AGAINST:
            keyword = "rebutt against"

        prompt = f"Here is the topic: {topic}. Given your expertise and document knowledge, generate a {keyword} {idea}. Be Concise."
        response = self.generate(prompt, self.assis_id, self.thread_id)
        self.write_log(f"Arg: {self.side}")
        self.write_log(response)
        self.write_to_csv(f'Arg: {self.side}')
        self.write_to_csv(response)
        return response
    
    def rebut_arg(self, topic, idea, arg):
        ''' Rebut argument '''
        prompt = f"Given the topic {topic} and the idea of {idea}, you operate on the {self.side} side of the debate. Generate a creative unique rebuttal \
                  using your expert knowledge in order to rebut the following argument. Ensure it is concise. Argument: {arg}"
        response = self.generate(prompt, self.assis_id, self.thread_id)
        self.write_log(f"Rebuttal: {self.side}")
        self.write_log(response)
        return response

    def idea_gen(self, topic):
        ''' Generate ideas '''
        prompt = f"Given the topic: {topic} generate a unique, interesting idea drawing upon your expert knowledge, perspective and uploaded documents."
        idea = self.generate(prompt, self.assis_id, self.thread_id)
        self.write_log("Idea:")
        self.write_log(idea)
        self.write_to_csv('Idea:')
        self.write_to_csv(idea)
        print(idea)
        return idea
        
if __name__ == "__main__":
    # Create the council and begin the idea generating process
    topic = input("Topic of interest: ")
    council = Council(topic)
    council.gen_ideas()
