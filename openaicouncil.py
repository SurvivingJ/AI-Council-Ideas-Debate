from openai import OpenAI
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
    def __init__(self, topic, model):
        # Determine what model and tools will be used
        if model == '':
            self.model = "gpt-4-1106-preview"
            self.extras_enabled_bool = True
        else:
            self.model = model
            self.extras_enabled_bool = False

        # API Key
        self.api_key = os.environ.get("OPENAI_AI_COUNCIL_KEY")
        if self.api_key is None:
            raise ValueError("Secret key not set in environment variables.")
        self.client = OpenAI(api_key=self.api_key)
        
        self.current_ideas = []

        # Topic
        self.topic = topic

        # Filter Keyword(s)
        self.keyword_and_cond = True
        keywords=self.filter_members()

        # Sentiment Analysis
        nltk.download('vader_lexicon')
        
        # Spawn members
        self.members = {}
        assistants = self.member_creation(self.model, keywords)
        for assistant in assistants:
            for_thread = self.client.beta.threads.create()
            opp_thread = self.client.beta.threads.create()
            self.members[assistant.id] = {'assistant': assistant, 'for_agent': CouncilMember(assistant, for_thread, Side.FOR), 'opp_agent': CouncilMember(assistant, opp_thread, Side.AGAINST)}
        
        # Spawn judges
        self.judges = []
        for _ in range(1):#####
            thread = self.client.beta.threads.create()
            judge_assistant = self.member_creation(self.model, ['judge'])
            judge = Judge(judge_assistant[0], thread)
            self.judges.append(judge)
    
    def filter_members(self):
        ''' Determine what tags to filter by '''
        with open('master_tags.txt', 'r') as f:
            tags = [tag.strip() for tag in f.read().split(',')]
        
        while True:
            filter_opt = input("Would you like to filter by requiring members to have all tags? Repond y or n ")

            if filter_opt == 'y':
                self.keyword_and_cond = True
                break
            elif filter_opt == 'n':
                self.keyword_and_cond = False
                break
            else:
                print("Invalid Input")
        
        print("Here is a list of all available tags")
        for i, tag in enumerate(tags):
            print(f"{i}: {tag}")

        selected_opts = []
        while True:
            selected_opt = input("Which tags would you like to filter by? Please type the number")

            if selected_opt not in selected_opts:
                try:
                    selected_opts.append(int(selected_opt))
                except:
                    print("Please respond with a number")
                    continue
                cont = input("Would you like to add more? Respond y or n ")
                if cont == "y":
                    continue
                else:
                    break
        
        selected_tags = []
        for opt in selected_opts:
            selected_tags.append(tags[opt])
        
        return selected_tags

    def member_creation(self, model, keywords = []):
        ''' Create an OpenAI Assistant based on information in the selected member's folder '''
        assistants = []
        try:
            # Get all entries in the directory
            entries = os.listdir('CouncilMembers')

            # Filter out and keep only folders
            folder_names = [os.path.join('CouncilMembers', entry) for entry in entries if os.path.isdir(os.path.join('CouncilMembers', entry))]
        except FileNotFoundError:
            # If the directory is not found, print an error message and return an empty list
            print(f"Directory CouncilMembers not found.")
            return []
        
        for folder in folder_names:
            instructions, file_names = self.read_assistant_data(folder, keywords)
            if instructions:
                if model == "gpt-4-1106-preview":
                    member = self.create_assistant(instructions, file_names)
                else:
                    member = self.create_assistant(instructions, file_names, tools=[], model=model)
                assistants.append(member)
        
        return assistants
    
    def read_assistant_data(self, directory, keywords):
        ''' Read Assistant data from relevant folders '''
        # Step 1: Read 'info.txt' to check for the keyword
        try:
            with open(f'{directory}/info.txt', 'r') as file:
                info_content = file.read()
        except FileNotFoundError:
            print("The file 'info.txt' was not found.")
            return None, []

        # Step 2: Check if the keyword is in 'info.txt'
        if self.keyword_and_cond:
            if all(keyword in info_content for keyword in keywords):
                # Step 3: Extract text from 'information.txt'
                try:
                    with open(f'{directory}/instructions.txt', 'r') as file:
                        instructions = file.read()
                except FileNotFoundError:
                    print("The file 'information.txt' was not found.")
                    return None, []
            elif len(keywords) == 0:
                # Step 3: Extract text from 'information.txt'
                try:
                    with open(f'{directory}/instructions.txt', 'r') as file:
                        instructions = file.read()
                except FileNotFoundError:
                    print("The file 'information.txt' was not found.")
                    return None, []
            else:
                print(f"Keywords '{keywords}' not found in 'info.txt'.")
                return None, []
        else:
            if any(keyword in info_content for keyword in keywords):
                # Step 3: Extract text from 'information.txt'
                try:
                    with open(f'{directory}/instructions.txt', 'r') as file:
                        instructions = file.read()
                except FileNotFoundError:
                    print("The file 'information.txt' was not found.")
                    return None, []
            elif len(keywords) == 0:
                # Step 3: Extract text from 'information.txt'
                try:
                    with open(f'{directory}/instructions.txt', 'r') as file:
                        instructions = file.read()
                except FileNotFoundError:
                    print("The file 'information.txt' was not found.")
                    return None, []
            else:
                print(f"Keywords '{keywords}' not found in 'info.txt'.")
                return None, []
            
        # Step 4: Create a list of file names in the specified directory
        try:
            file_dir = directory + '/Files'
            file_names = [os.path.join(file_dir, file) for file in os.listdir(file_dir)]
        except FileNotFoundError:
            print(f"Directory '{directory}' not found.")
            return instructions, []
        
        return instructions, file_names
        
    def create_assistant(self, instructions, files, tools=[{"type": "retrieval"}], model="gpt-4-1106-preview"):
        ''' OpenAI Assistant creation API calls '''
        if self.extras_enabled_bool:
            # Upload a file with an "assistants" purpose
            file_ids = []
            for file in files:
                file = self.client.files.create(
                file=open(file, "rb"),
                purpose='assistants'
                )
                file_ids.append(file.id)

            # Create an assistant using the file ID
            assistant = self.client.beta.assistants.create(
            instructions=instructions,
            model=model,
            tools=tools,
            file_ids=file_ids
            )
        else:
            assistant = self.client.beta.assistants.create(
            instructions=instructions,
            model=model
            )
        return assistant

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
        print("================")
        print("BEST OPTION:")
        print(best_idea_str)
        print("================")

    def wait_on_run(self, run, thread):
        ''' Wait for response from OpenAI API '''
        while run.status == "queued" or run.status == "in_progress":
            run = council.client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            time.sleep(0.5)
        return run

    def submit_message(self, assistant, thread, user_message):
        ''' Submit message to OpenAI API '''
        council.client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=user_message
        )
        return council.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )
    
    def get_response(self, thread):
        ''' Get response from OpenAI API '''
        return council.client.beta.threads.messages.list(thread_id=thread.id, order="asc")
    
    def parse_response(self, response):
        ''' Parse OpenAI API response '''
        messages = []
        for thread_message in response.data:
            for content in thread_message.content:
                    message_text = content.text.value
                    messages.append(message_text)
        return messages
    
    def generate(self, prompt, assistant, thread):
        ''' Model call to OpenAI '''
        run = self.submit_message(assistant, thread, prompt)
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
    def __init__(self, assistant, thread):
        self.scores = []
        self.assistant = assistant
        self.thread = thread

    def pass_judgement(self, topic, idea, for_arg, opp_arg):
        ''' Pass judgement on the quality of the idea '''
        self.write_log(f"ARGS: forarg -> {for_arg} || opp_arg -> {opp_arg}")
        for_prompt = f"Pass judgement on the following argument. The topic is {topic} and the idea is {idea}. \
                  The for argument is {for_arg}."
        opp_prompt = f"Pass judgement on the following argument. The topic is {topic} and the idea is {idea}. \
                  The opp argument is {opp_arg}."

        for_response = self.generate(for_prompt, self.assistant, self.thread)
        opp_response = self.generate(opp_prompt, self.assistant, self.thread)

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
    def __init__(self, assistant, thread, side):
        self.side = side
        self.assistant = assistant
        self.thread = thread

    def create_arg(self, topic, idea):
        ''' Create arguments for the idea '''
        if self.side == Side.FOR:
            keyword = "argue for"
        elif self.side == Side.AGAINST:
            keyword = "rebutt against"

        prompt = f"Here is the topic: {topic}. Given your expertise and document knowledge, generate a {keyword} {idea}. Be Concise."
        response = self.generate(prompt, self.assistant, self.thread)
        self.write_log(f"Arg: {self.side}")
        self.write_log(response)
        self.write_to_csv(f'Arg: {self.side}')
        self.write_to_csv(response)
        return response
    
    def rebut_arg(self, topic, idea, arg):
        ''' Rebut argument '''
        prompt = f"Given the topic {topic} and the idea of {idea}, you operate on the {self.side} side of the debate. Generate a creative unique rebuttal \
                  using your expert knowledge in order to rebut the following argument. Ensure it is concise. Argument: {arg}"
        response = self.generate(prompt, self.assistant, self.thread)
        self.write_log(f"Rebuttal: {self.side}")
        self.write_log(response)
        return response

    def idea_gen(self, topic):
        ''' Generate ideas '''
        prompt = f"Given the topic: {topic} generate a unique, interesting idea drawing upon your expert knowledge, perspective and uploaded documents."
        idea = self.generate(prompt, self.assistant, self.thread)
        self.write_log("Idea:")
        self.write_log(idea)
        self.write_to_csv('Idea:')
        self.write_to_csv(idea)
        print(idea)
        return idea
        
if __name__ == "__main__":
    # Create the council and begin the idea generating process
    topic = input("Topic of interest: ")
    #model = input("What model will you use? ")
    model = 'gpt-3.5-turbo'
    council = Council(topic, model)
    council.gen_ideas()
