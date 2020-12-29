#!/usr/bin/env python
# coding: utf8
"""Example of training an additional entity type

This script shows how to add a new entity type to an existing pretrained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.

After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities
en_core_web_md
Compatible with: spaCy v2.1.0+
Last tested with: v2.2.4
"""
from __future__ import unicode_literals, print_function

import plac
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import pandas as pd


# new entity label
LABEL = "TOPIC"
#train_data = pd.read_excel("annotated.xlsx")


# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
TRAIN_DATA = [
    (
        "But will the current bans last? Dave Gershgorn Dec 18Â·4 min read OneZeroâ€™s General Intelligence is a roundup of the most important artificial intelligence and facial recognition news of the week. The facial recognition industry has been quietly working alongside law enforcement, military organizations, and private companies for years, leveraging 40-year old partnerships originally centered around fingerprint databases. But in 2020, the industry faced an unexpected reckoning. February brought an explosive New York Times report on Clearview AI, a facial recognition company that had scraped billions of images from social media to create an all-encompassing database, and quietly gave it to thousands of police departments and companies across the world. A ubiquitous facial recognition database weaponizing public social media profiles to create tools for law enforcement and private security was a splash of cold water for those who hadnâ€™t yet understood what the facial recognition industry had become. Now the technology has never been easier to implement by corporations and law enforcement, and the ramifications of the technology have never been more serious. Across the United States, bans on police use of facial recognition had started in 2019 with San Francisco. But in 2020 they proliferated, thanks to activist pressure from organizations like the ACLU, Fight For The Future, and AI Now. Portland, Oregon, instituted one of the strictest facial recognition laws in the world, and by far the most restrictive in the country, which prohibits private businesses from using the technology on their own premises. This means businesses like Walmart wouldnâ€™t be able to use facial recognition to track customers in their stores. â€œThis is really about making sure that we are prioritizing our most vulnerable community members and community members of color,â€ Portland City Council Commissioner Jo Ann Hardesty said in Kate Kayeâ€™s report earlier this year for OneZero. Portland, Maine, also banned police use of the technology this year, alongside bans and temporary moratoriums in Madison, Wisconsin; Springfield, Massachusetts; Cambridge, Massachusetts; Jackson, Mississippi, and Boston. Even Los Angeles placed a limited facial recognition ban on its police department, which prevented it from using facial recognition that sources images from social media, like Clearview AI. The LAPD will still use countywide facial recognition software that references booking images, according to BuzzFeed News. â€œThis is really about making sure that we are prioritizing our most vulnerable community members and community members of color.â€ While Massachusetts legislators passed a statewide ban on police use of the technology, Governor Charlie Baker has reportedly refused to sign the bill into law, claiming that facial recognition is a necessary tool for police. This comes after players of the Boston Celtics basketball team wrote an open letter to Baker, asking him to sign the bill. â€œBakerâ€™s rejection is deeply troubling because this technology supercharges racial profiling by police and has resulted in the wrongful arrests of innocent people,â€ the players wrote. The players are referring to a case in Detroit, where Robert Julian-Borchak Williams was wrongfully arrested due to an error in the policeâ€™s facial recognition system. The federal government has also taken notice of the rise of facial recognition, with renewed pressure from Congress to legislate the technology. Democrats introduced a bill that would ban federal agencies from using facial recognition, and withhold federal funds from municipalities unless they also enacted such bans. This bill came weeks before an updated report from the Government Accountability Office, which recommended facial recognition legislation to safeguard citizensâ€™ privacy. Even tech giants like IBM, Amazon, and Microsoft, which are some of the largest creators and proponents of the technology in the consumer space, have stopped selling the technology to police, and, in IBMâ€™s case, canceled facial recognition development altogether. Legal challenges filed in 2020 against Clearview AI and the use of facial recognition in schools are scheduled to be decided in 2021, according to a report from CNETâ€™s Alfred Ng. Clearview AI is facing lawsuits in Illinois, which has the countryâ€™s best protections against companies secretly collecting fingerprints, DNA, and facial images. These legal battles could set precedent for how facial recognition companies are allowed to collect data and act as a call to action for other states as well. All of this movement to regulate and restrict the use of facial recognition flies in the face of the enormous corporate and governmental interest in being able to identify people in public and private spaces. Marketers have long pitched facial recognition and other identifying technologies as the ultimate tool for understanding customers. And police departments often sing the praises of being able to pick out the identity of a suspect from a photo or video. That makes the activistsâ€™ progress in 2020 all the more impressive, and the stakes all the higher. In 2021, Amazonâ€™s self-imposed facial recognition moratorium is set to expire, opening up questions as to whether it will actually commit to stop its work with facial recognition or continue working with police, as it already has with its Ring smart doorbell. Microsoft and IBM have also sent letters to President-elect Biden, asking for national facial recognition laws. Those laws could curb the use of the technology â€” or add cover for the use of facial recognition by law enforcement in ways we can hardly imagine.",
        {"entities": [(32, 46, "PERSON"),(134, 157, LABEL), (162, 180, LABEL), (283, 305, "ORG"), (622, 634, LABEL), (1187, 1200, "GPE"), (1268, 1281, "GPE"), (1402, 1404, LABEL), (1657, 1664, "ORG"), (2038, 2048, LABEL), (2217, 2228, "GPE"), (2370, 2381, LABEL), (2917, 2923, "GPE"), (3252, 3282, "PERSON"), (3850, 3857, LABEL), (3881, 3884, "ORG"), (3886, 3892, "ORG"), (3898, 3907, "ORG"), (3988, 3998, LABEL), (4462, 4468, LABEL)]},
    ),
    ("AI Platform Notebooks provide managed JupyterLab notebook instances, a familiar tool to experiment, develop, and deploy models into production. The missing piece is training models for production; this typically runs much longer than the initial experimentation or the production prediction. The interactive Notebook in which you did your experimentation is probably not the right place for this: browser sessions time out, connections are lost, image rendering freezes, and the instance isnâ€™t right-sized for training. This article focuses on two approaches to running your training in a headless manner: AI Hub JupyterLab VM: Papermill AI Platform Training AI Hub JupyterLab VM: Papermill Papermill is a tool for parameterizing and executing Jupyter Notebooks in a number of different ways including by spinning up a VM. One of these is submitting the notebook from the command line of the Notebookâ€™s VM: this takes advantage of the custom VM configuration which youâ€™ve already configured with your AI libraries, while bypassing the potential freezing issues associated with rendering the results in the Notebook UI. The Notebook VM also includes Papermill pre-installed, so no additional configuration is required. If youâ€™d like to run on a separate VM with Papermill, this TPU-based Next 2019 talk walks through that scenario. You can access the command line either from the Cloud Console -> Compute Engine -> VM instances -> SSH option for the VM, or from within the Notebook itself; the former excludes the Notebook UI as a potential concern. You can open the SSH session in a browser window, or in a dedicated client for additional state persistence. Parameterization You can parameterize your Notebook to pass variable information such as the number of epochs into the notebook at run time. Set default parameters in a cell tagged with the â€œparameterâ€ keyword and refer to these in the training or other steps; these are then over-ridden by whatever you specify on the command line. Eg. Command Line for a Notebook with an epochs parameter: Execution When you run the command, you specify an output Notebook; all intermediate results, logging, etc. are recorded here. Include the log-output flag to write notebook output to stderr (ie. the terminal window) Note that if the terminal is closed, this terminates the SSH session, and by default any Notebooks running within it, including headless notebooks. Continue execution if the terminal is closed To continue execution even if the terminal session is closed, use a command sequence like one of the following: These commands show the process id (pid) and redirect stderr and stdout to nohup.out or ~/output.txt respectively. You can monitor the running process from from the output notebook, as well as from a new terminal session using the following command: Note that redirecting stdout and stderr throws the following error; this doesnâ€™t affect Notebook execution or logging: AttributeError: â€˜NoneTypeâ€™ object has no attribute â€˜send_multipartâ€™ More background here: Running commands which survive terminal close Difference between nohup, disown, and & AI Platform Training This approach is based on creating a Docker image with your libraries layered on top of a base Tensorflow image, then deploying that to AI Platform Training; you can also run the image locally for testing. This has the advantage of being able to size your training environment differently from your Notebook environment, and natively close the terminal session without affecting job execution; you can track job execution in the cloud console; logs are written to Stackdriver. Store the model weights in Google Cloud Storage as the link between the training and prediction steps; note that you can also save the whole model instead of just the weights; the chosen approach will depend on your use case. Parameterization As before, you can parameterize the execution, eg. with the number of epochs. Sample script Execution Step through the steps outlined above on your VM, or run it all as a bash script. You will need to create your own GCS bucket in the project in which youâ€™re running the VM; this GCS bucket can be on a different VM if you configure service accounts appropriately. If you choose to run the optional local training test step, your VM will need a GPU. Checkpoints â€‹To support better recovery from failures during a job, you can implementâ€‹ checkpoints; this Keras sample demonstrates checkpointing on epochs.",
     {"entities": [(0, 11, LABEL), (132, 142, LABEL), (113, 126, LABEL), (100, 107, LABEL), (280, 290, LABEL), (296, 307, LABEL), (652, 660, LABEL), (1422, 1434, LABEL), (3662, 3674, LABEL), (3675, 3682, LABEL)]}),
    (
        "Analytics are too tall and they pretend to care about your feelings like data entry",
        {"entities": [(0, 9, LABEL), (73, 83, LABEL)]},
    ),
    (
        "Who is Shaka Khan?",
        {"entities": [(7, 17, "PERSON")]},
    ),
    (
        "I like London and Berlin.",
        {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]},
    ),
    (
        "As hard as I tried, I couldnâ€™t shake off the thought of another stream of income meandering into my cash pool. The excitement bubbling up within my mind swallowed all the doubts I had. High on excitement and big on promise, I dove head-first into the side hustle hole. That was in mid-2017 when the spark of the flash lured me to the photo studio. I wanted to be like a colleague who shot weddings every other weekend. He persuaded me I could even divert into architectural photography, which, for a real estate surveyor, was music to my ears. In 2019, a colleague suggested we co-authored a peer-reviewed article. I didnâ€™t know any better than to offload rounds of rapid-fire follow-up emails to the editorial team. Judging by the echoing silence I got in response, I guess most of them went into the trash folder. Dejected, I realized I could channel the frustration into writing articles. Between starting a WordPress blog and writing on Medium, the pen is proving to be a powerful weapon, if not quite mightier than the sword just yet. As the year trudges to the finish line, I sat down to count my blessings and recount the lessons Iâ€™ve learned combining two side-hustles with my career in real estate. They were bountiful, thanks. But here are only three. The more you have on your plate, the more time you can create When I thought of starting a photography business on the side of my grueling work schedule as an Estates Assistant, part of me wondered whether I would have any time left to brush my teeth. But the siren call of more money was enough to convince me to at least give photography a shot. Even though I smiled at the first paycheck, my second career threatened to take up too much time if I didnâ€™t zoom in and focus on the fine details of my time allocation. So I did a time audit. It turned out the mindless scrolling on Twitter and Facebook, and my mini-addiction to watching every WhatsApp status took between two and four hours of my day. I knew if I focused less on my friendsâ€™ rants and strangersâ€™ stunts online, I could take back some of those hours. Adding another side hustle in writing meant I had to keep an even tighter rein on my time. Challenge accepted. Now, outside of my morning routines and breakfast, I try to write about two hours Monday to Wednesday and read for another hour before midday. My lunch break, mail time, and social media take over until around 3 p.m. A 20â€“30 minute nap follows. In the evenings, I keep a fluid schedule that mixes taking writing courses, doing research work, and reading articles online with spending time with the family. I try to schedule photo sessions and property inspections for Fridays, but depending on the clientsâ€™ demands, these could fall on any day. Iâ€˜ve scheduled my laundry for Saturdays, mostly while I listen to the Guardian Football Weekly podcast or a few NBA-related pods from ESPN. I am free to go to church and do a whole lot of non-work-related activities on Sundays. But most importantly, I do not starve myself of the recommended seven to nine hours of sleep. I stay on schedule about 60% of the time, but thatâ€™s a decent mark considering the chopping and changing I have had to make as my commitments change. Itâ€™s hard to speak for everyone because we are all different. For example, I donâ€™t have pets, kids, or extended family to cater for. And I have a partner who respects my schedules. But I have realized that the more work I have had on my hands, the more creative I have become with my time, leaving me with more time to spare. More work will make you better than you could imagine I always try to be the best in every endeavor I take up. This means I have had to read more pages, endure harder practice sessions, battle tough failures, and learn from those failures. But it turns out thatâ€™s the best route to a bountiful harvest from different fields. YouTube keeps feeding me with lots and lots of photography tutorials. And thanks to platforms like Edx and Coursera, Iâ€˜m taking courses to learn how to write better, how to communicate clearly. I am now aware of some basic rules of writing and communication, and I can see when a newbie breaks them. When my photography side business took off, I soon realized there was more to it than posing people and clicking shutters. I had to learn skills like sales, marketing, customer relations, and bookkeeping on the fly. Iâ€™ve also learned to network and support other writers â€” well more than what I thought writing was about. But when I look back, I realize I would not have learned most of these skills so fast if I was only on my first career as a real estate surveyor. I am happy with the progress Iâ€™m making. Similarly, when I look at a few of my colleagues who have also added different dimensions to the careers school prepared them for, I see how theyâ€™ve gotten better. Some have mastered public speaking, teaching, disk jockeying, etc. Yes, I can say I have gotten better and learned more skills from adding two side hustles to the career school prepared me for. Youâ€™d best ignore a lot of the grind and hustle advice If I had listened to some of the advice on hustling and grinding, I wouldâ€™ve ground myself into powder. â€œGrind, work hard, sleep less, work, even more, keep your nose to the grindstone and grind harder,â€ youâ€™ll hear some evangelists preach. A colleague even advised me to wake up an hour earlier, scrap the nap and sleep an hour later than my usual bedtime so I can steal two and a half hours from nature. I bought it, not knowing any better then. But it didnâ€™t take long for me to realize I was harming myself more than I was benefiting from the marginal gains I enjoyed. That was even before I read about some of the effects of the lack of adequate sleep. People have gone after Gary Vee for leading a grind and hustle culture that threatens to turn people into mules glistening with sweat. But when I listened to him, I realized that was not what he meant. In this YouTube video, he said hustling means going all-in and maximizing the 15 hours a day when heâ€™s awake. In another video, he says he takes between four and seven weeks of vacation in a year. Yes, some startup founders have had to work harder, sleep less, and take even fewer vacations, depending on their roles and goals. Iâ€™m no organizational psychologist, but I have also realized everyone doesnâ€™t require that level of work to be successful. Thankfully, Iâ€™ve not had to survive on four hours of sleep a night, as some ultra-hard working people do. Occasionally, a tight deadline pushes me past my 10 p.m. bedtime or takes up most of my day, but thatâ€™s normal. When I see the threat of overworking looming, I occasionally delegate roles like location scouting and photo editing. This frees up even more time for me to recharge my batteries. In short, trash a lot of the grind and hustle advice that says to work yourself into bare bones before you can taste the juicy fruits of your labor. Conclusion I may not be the archetypal entrepreneur who came up with a disruptive idea that turned the world upside down to make it better. People like that are the exception. But for me, when the side-hustle train â€” with lots of people hanging from the sides â€” curled into view, I hopped on and joined the party. No, none of them rakes in a million dollars in revenue yet, but that was never the goal. Along the way, I have learned a lot of invaluable lessons, the three most important being: 1. The more you do, the more time you can create. 2. The more you do, the better youâ€™d become. 3. People misunderstand a lot of the grind and hustle advice. Yes, you can run a small business or two on the side of your career and still have your body in one piece. Itâ€™s hard, and you may have to be super creative and innovative. But since when didnâ€™t you have to possess those qualities in the 21st century?",
        {"entities": [(476, 487, LABEL), (18, 24, "LOC")]},
    ),
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model="en_core_web_md", new_model_name="tech_startup", output_dir="output", n_iter=30):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(LABEL)  # add new entity label to entity recognizer
    # Adding extraneous labels shouldn't mess anything up
    #ner.add_label("VEGETABLE")
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_text = "We provide data entry services and Outsource Data Entry Services."
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":
    plac.call(main)