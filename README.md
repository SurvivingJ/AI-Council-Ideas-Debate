# AI Council - Idea Generation & Evaluation
A system of LLM based bots to generate ideas and subsequently evaluate their logic, rationality and cohesiveness, and determine the best idea.

# Idea Generation Methodology
## Background and Design
Large Large Models (LLMs) are a useful tool for idea generation, debate and evaluation. They are able to discuss an impossibly large range of topics compared to the average knowledge-base of their human counterparts. According to Girotra et al. (2023), “Two hundred ideas can be generated by one human interacting with ChatGPT-4 in about 15 minutes. A human working alone can generate about five ideas in 15 minutes.” Moreover, in that study, 35 of the top 40 ideas had been generated by ChatGPT-4. While the most novel ideas in the top 40 were those produced by humans, this disparity in quality of ideas cannot be ignored, nor the speed of generation. 

In the world of prompt engineering, it is also well known that the thought processes and inspirations that guide a LLM during discussions can be directed by imbuing them with a “simulacra”, whether that be a psychologist or Warren Buffet. As a result, not only can LLMs generate more ideas at a faster rate than humans, but they can create them using different fields, writings and philosophies as a basis for their perspective. This idea is important to exploit when attempting to gain the highest-quality and most specific responses from a LLM.

For centuries it has been commonly known that academic progress comes from the debate and in-depth evaluation of ideas. As a result, I decided to design a “Council” of agents that could generate, debate and judge the quality of ideas. While originally intended for Instrumental Variable idea generation, this “Council” could be used for generating ideas for anything.

The Council is composed of Members and Judges. Each Member is imbued with a simulacra, from “John Maynard Keynes” to “Warren Buffet”, and has the ability to generate ideas, arguments for, and arguments against the use of a trading strategy. The Judges are told to analyse how rational and logical the arguments are from each side and present a judgement, which is then analysed using sentiment analysis. I then rank the strength of each idea based on the overall sentiment scores, and pick the best idea.

The speed at which relatively strong debate on the generated ideas was conducted was impressive, and far beyond the realm of human capabilities, even when the Council program, as well as LLM processing speeds and costs, are at their most primitive, relative to their future potential.

## Assistants
With the introduction of Assistants, many of the planned improvements have become infinitely easier. Users can choose to use whichever model they want, however the choices are essentially between ‘gpt-4-1106-preview’ and ‘gpt-3.5-turbo’. Regardless of model choice, a new instance of each Assistant is created, they are prompted to guide their ‘simulacra’, and then these Assistants are accessible as objects in a dictionary. Each ‘For’ and ‘Against’ Council member will have its own memory of ideas it generated and arguments it has made, hopefully reducing repetition in their own arguments. If ‘gpt-4-1106-preview’ is chosen, then select Council members are also given a knowledge base which they can draw upon. For “Warren Buffet” it is a compilation of his shareholder letters, as well as the first edition of The Intelligent Investor by Benjamin Graham, while for John Maynard Keynes, it is his “The General Theory of Employment, Interest, and Money”. As the financial cost and efficiency of these services move quickly in a more enabling direction, eventually we may be able to use extremely high quality simulacras in all runs of the Council.

## Limitations
The main limitation of the implementation of this idea was both API call costs, as well as API request throttling from OpenAI, which initially limited the number of agents I could use and the number of ideas I could evaluate [**This seems no longer to be a problem with the advent of Assistants, as I do not call to each Council Member often, but initially was making every call through the Chat Completion point**]. I also experimented with using smaller, local LLMs such as the GPT4All Falcon model. However, while the quality of the responses were noticeably, but not remarkably, different, it was the extremely long processing time per response that inhibited my use of a free local model. Instead of the 10-15 minutes it took using OpenAI, it is estimated that it would have taken us at minimum a day. In short local tests, however, I found that the extended processing time occasionally led to timeout errors, and thus a day-long test would likely have been impossible to perform. Furthermore, the primitive nature of the program means that I have not been able to obtain as much value from the Council as I may in the future. [**Moreover, the introduction of Assistants is a massive step forward in the capabilities of the Council, which I do not know of being available (as of yet) in GPT4All**].

While there appears to be some increase in the quality of output from using the ‘gpt-4-1106-preview’ model, for USD$30 I was unable even to finish a run that took me USD$2 using ‘gpt-3.5-turbo’. Additionally, the whole ‘gpt-3.5-turbo’ run finished significantly quicker than ‘gpt-4-1106-preview’ would have, finishing even before I stopped the more advanced model. Thus, given disparities in financial cost and efficiency, I recommend using the ‘gpt-3.5-turbo’ model. While the ideas generated or arguments made may not be as high in quality, the ‘gpt-3.5-turbo’ model is still excellent to serve the Council’s purpose.

Another limitation of using the ‘gpt-4-1106-preview’ model is the need for written work by the Member to use as a knowledge base. With many incredible researchers and thinkers still alive, their work has not entered public domain, and is often locked behind paywalls, which is another limitation of using the advanced model for the Council at this time.

## Future Improvements
This idea generation and evaluation “Council” model is only a basic adaptation and implementation of an idea originally intended for Instrumental Variable idea generation in econometric analysis. In future iterations, I will look to:
- ~~Implement memory for the agents in order to minimise repetition of similar ideas and allow them to use previous discussion as a springboard for rebuttals and arguments~~
- Finetune prompts, temperature and other parameters related to the LLM response generation [**Progress Made**]
- Expand the Member templates so that an inbuilt array of “social simulacra” are available and already finetuned [**Progress Made**]
- Introduce functionality so that one can filter based on field or person, and overall utilise a greater range of perspectives [**Can now filter based on field**]
- Create a clearer structure and cleaner data output

## Future
There is still a lot of improvements to be made, both in Council design, as well as Council Member development, and thus any and all contributions are welcome.

## A Few Random Examples (formatted by me, but text unchanged)
### Topic
Mechanism for encouraging greater citizen participation at a local council level

### Best Idea gpt-turbo-3.5
Drawing upon Keynesian economics, one unique and interesting idea to give the population greater control over decision making at a local council level would be to implement a participatory budgeting system. This concept aligns with Keynes's emphasis on government intervention and the importance of aggregate demand in economic management.

Participatory budgeting involves directly involving community members in the decision-making process for allocating public funds. This process allows citizens to have a say in how their tax dollars are spent and gives them greater control over local economic policies. It also promotes transparency, accountability, and inclusivity in governance.

In this system, the local council would hold public meetings where citizens can propose and discuss potential projects, programs, or services that require public funding. These proposals would then be evaluated by experts to determine their feasibility, cost, and potential impact. The council would provide information on available funds and any budgetary constraints.\n\nNext, the citizens would vote on the proposed projects or services to prioritize funding allocation. The voting process could be conducted online, through mobile applications, or in-person voting booths, ensuring accessibility for all members of the community. The projects with the highest number of votes would receive funding until the allocated budget is utilized.

By implementing participatory budgeting, citizens would have the opportunity to directly influence local economic policies and shape their communities according to their needs and preferences. This not only empowers individuals but also encourages active civic engagement and fosters a sense of ownership and responsibility within the community.

Keynes's macroeconomic principles support this idea by recognizing the importance of aggregate demand. When communities have a voice in determining the allocation of public funds, they can prioritize projects that stimulate local demand, create jobs, and foster economic growth. Additionally, participatory budgeting aligns with Keynes's view on government intervention, as it encourages active government involvement in economic decision making, enabling policymakers to better understand the needs and expectations of the population.

Overall, implementing a participatory budgeting system at the local council level can create a more inclusive and democratic approach to economic management, giving the population greater control and involvement in shaping their communities.

### Idea Comparison
___
### Adam Smith ‘gpt-turbo-3.5’
Market-Based Decision Making in Local Councils

#### Idea Overview:
- Implementing a market-based system in local councils, drawing upon Adam Smith's principles of free markets and the 'invisible hand' concept. This approach would entail:
- Establishing Competition: Setting rules to promote competition among individuals and businesses proposing community services or policy changes.
- Proposal Submission: Parties submit plans addressing community needs.
- Public Evaluation: Residents review, give feedback, and express preferences on proposals.
- Council's Role: Facilitate transparency, ensure fair competition, and monitor outcomes.
- Community-Driven Decisions: Residents influence selection based on needs and preferences, fostering innovation and service quality.
- Aligning with Adam Smith: Motivation through self-interest, division of labor, and limited government intervention.
- Government's Role: Identifying areas for market mechanisms and necessary interventions.

#### Conclusion:
This system enhances residents' control in decision-making, promoting competition, efficiency, and satisfaction of community needs.

### Adam Smith ‘gpt-4-1106-preview’
#### 1. The Democratic Marketplace
#### Concept Overview:
The Democratic Marketplace is an online platform inspired by Adam Smith's free market principles. It incentivizes democratic participation by allowing individuals to invest "Civic Coins" into local initiatives, policies, or candidates they support.

#### Mechanics:
- Acquisition of Civic Coins:
- Earned through participation in democratic processes like attending town meetings, voting, community service, or policy discussions.
- Investment and Dividend:
- Invest Civic Coins in local initiatives or candidates. Dividends are earned if initiatives succeed or candidates win.

#### Marketplace Dynamics:
- The success of initiatives or candidates affects the investment value, encouraging active, constructive local governance engagement.

#### Redeeming Value:
- Redeem Civic Coins for benefits like local service discounts, community facility bookings, or tax credits.

#### Transparency and Education:
- The platform provides comprehensive information on local issues to inform investments and foster a well-informed electorate.

#### Alignment with Adam Smith's Principles:
- Self-Interest: Personal incentives for community benefit.
- Division of Labor: Specialization in governance areas for Civic Coins.
- Limited Government Intervention: Empowers citizens in governance with minimal top-down regulation.
- Competition and Efficiency: Promotes support for viable, beneficial proposals.

#### Challenges:
- Equal access to Civic Coins.
- Preventing manipulation.
- Supplementing, not replacing, other democratic engagements.
___
### Alfred Marshall ‘gpt-3.5-turbo’
Digital Platform for Direct Participatory Democracy

#### Concept:
A digital platform for participatory democracy, influenced by Marshall's economic theories, enabling:
- Citizen Expression: Voice opinions and shape local policies.
- Incorporating Elasticity of Demand: Reflect varying intensity of citizen preferences.
- Proposal Submission and Discussion: Residents suggest and debate local initiatives.
- Weighted Voting: Based on expertise, addressing power imbalances.
- Tracking Support: Gauge demand elasticity for ideas, maximizing community welfare.
- Technology Integration: Enhances engagement, transparency, and accountability.

#### Implication:
Empowers citizens, aligns with Marshall's welfare economics, and ensures well-being of the local population.

### Alfred Marshall ‘gpt-4-1106-preview’
#### 2. Democratic Participation Market (DPM)
An economic system directly tied to democratic engagement, applying Alfred Marshall's theories. DPM uses supply and demand principles where citizens earn 'participation credits' through democratic activities. Credits can be redeemed for benefits like tax credits or public service discounts. The market’s supply side is controlled by local authorities, and demand comes from residents. Elasticity adjusts the credit value to maintain market equilibrium.
___
### Amos Tversky ‘gpt-3.5-turbo’
#### Citizen's Decision Lab
#### Concept:
A lab for collaborative decision-making, drawing from Amos Tversky's research:
- Random Selection: Diverse citizen representation.
- Bias Training: Educate on cognitive biases.
- Deliberative Discussions: Facilitate critical thinking and collective decision-making.
- Decision Support Tools: Aid in systematic evaluation.
- Iterative Process: Encourage considering alternatives.
- Transparency: Public access to discussions and decisions.

#### Implications:
Overcomes rational choice model limitations, ensuring informed community representation.

### Amos Tversky ‘gpt-4-1106-preview’
#### 4. Gamified Civic Engagement Platform
A platform integrating gamification elements (points, badges, leaderboards) to encourage civic participation, based on Tversky’s research. It addresses cognitive ease, loss aversion, and other heuristics to motivate involvement in activities like town meetings or elections.
___
### John Maynard Keynes ‘gpt-3.5-turbo’
#### Participatory Budgeting Inspired by Keynesian Economics
#### Proposal:
A participatory budgeting system aligning with Keynes's government intervention and aggregate demand theories:
- Public Meetings: Discuss potential projects and services.
- Expert Evaluation: Assess feasibility and impact.
- Citizen Voting: Determine funding priorities.
- Economic Impact: Stimulate local demand and growth.

#### Outcome:
More inclusive, democratic economic management, and community shaping.

### John Maynard Keynes ‘gpt-4-1106-preview’
#### 8. Democratic Investment Credits (DICs)
A Keynesian-inspired system where individuals earn DICs through democratic activities, investing in local funds and receiving dividends from community improvements. This links civic engagement with personal economic well-being.
___
### Milton Friedman ‘gpt-3.5-turbo’
#### Blockchain-Based Digital Voting System for Local Councils
#### Proposal:
Implement a blockchain digital voting system for transparent, secure local decision-making:
- Digital Identity Verification: Ensure eligible resident participation.
- Proposal Generation: Submit and review local initiatives.
- Transparent Evaluation: Open review of proposals.
- Decentralized Voting: Blockchain technology for secure voting.
- Real-time Feedback: Ongoing community updates and involvement.

#### Alignment:
Resonates with Friedman's principles of individual freedom and limited government intervention.

### Milton Friedman ‘gpt-4-1106-preview’
#### 18. Democratic Participation Dividend
A Friedman-inspired approach offering CivicCoins for democratic participation. It includes a marketplace for redemption, volatility management, and tax incentives for businesses supporting local democracy.