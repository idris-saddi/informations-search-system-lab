import re
from typing import Dict, Iterable, List, Set, Mapping

# 1. Creation du corpus (inspire des slides 14 du mod. langue et 16 du mod. vectoriel)
documents = {
    "d1": """
    Jackson was one of the most talented entertainers of all time, boasting a career that spanned over four decades and reshaped the landscape of popular culture. 
    His incredible range of artistic abilities included not just a distinct and powerful vocal style, but also groundbreaking dance moves and visionary songwriting. 
    From his early days as the child prodigy fronting The Jackson 5 to his monumental success as a solo artist, he consistently pushed the boundaries of what was possible in music and performance. 
    His album 'Thriller' remains a global phenomenon, holding the record for the best-selling album in history, a testament to his universal appeal. 
    Beyond the sales figures, his talent was evident in his ability to merge genres, blending pop, soul, rock, and funk into a sound that was uniquely his own. 
    His live performances were legendary spectacles, setting new standards for touring artists with their elaborate choreography, special effects, and sheer energy. 
    Critics and fans alike lauded his ability to connect with audiences on an emotional level, making him a true icon. 
    Even years after his passing, his influence can be seen in the work of countless modern artists who emulate his style, work ethic, and commitment to the art of entertainment. 
    He was not merely a singer; he was a complete performer whose talent transformed the industry forever.
    """,

    "d2": """
    Michael Jackson anointed himself King of Pop, a title that, while self-proclaimed, was widely accepted by the public and the music industry due to his overwhelming dominance in the 1980s and 1990s. 
    This moniker reflected his unparalleled success on the charts, his massive global fan base, and his profound influence on the direction of popular music. 
    Elizabeth Taylor famously used the term when introducing him at an awards ceremony, cementing its place in the public lexicon. 
    Being the King of Pop wasn't just about selling records; it was about defining the visual and sonic aesthetic of an era. 
    He turned music videos into short films, transforming the medium from a promotional tool into a respected art form with masterpieces like 'Billie Jean', 'Beat It', and 'Thriller'. 
    His fashion choices—the single glove, the fedora, the military jackets—became iconic symbols recognized instantly around the world. 
    His ability to cross racial and cultural barriers was significant, as he became one of the first African American artists to receive heavy rotation on MTV, paving the way for future generations. 
    The title 'King of Pop' encapsulates a legacy of innovation, showmanship, and a level of superstardom that few, if any, have managed to replicate since his reign.
    """,

    "d3": """
    Car insurance and auto insurance are critical financial safety nets designed to protect drivers from the staggering costs associated with vehicle accidents and ownership. 
    While the terms are often used interchangeably, they refer to a contract between the vehicle owner and an insurance company where the owner pays a premium in exchange for financial protection against losses. 
    This protection is vital because a single accident can result in medical bills, repair costs, and legal fees that could easily bankrupt an individual. 
    Most jurisdictions mandate a minimum level of liability coverage to ensure that if a driver injures someone else or damages their property, there are funds available to cover those damages. 
    However, auto insurance goes beyond just liability; it can also provide coverage for the policyholder's own vehicle through collision and comprehensive policies. 
    Collision coverage pays for damage to the car resulting from a crash, while comprehensive coverage protects against non-collision events like theft, vandalism, fire, or natural disasters. 
    Navigating the world of car insurance involves understanding these different components, including deductibles and policy limits, to ensure that one is adequately protected against the myriad risks of the road.
    """,

    "d4": """
    Finding the best car insurance involves balancing cost, coverage options, and customer service reliability to suit individual needs. 
    There is no single "best" insurer for everyone because rates are highly personalized, based on factors such as age, driving history, location, and the type of vehicle being insured. 
    For some, the best insurance is simply the cheapest option that meets legal requirements, allowing them to drive legally without breaking the bank. 
    For others, the best insurance implies a premium service with comprehensive coverage, low deductibles, and perks like roadside assistance and accident forgiveness. 
    Evaluating insurance companies requires looking beyond the price tag; one must consider the insurer's financial stability and their reputation for handling claims efficiently and fairly. 
    Reading customer reviews and checking ratings from independent agencies can provide insight into how a company treats its policyholders during stressful times, such as after an accident. 
    Ultimately, the best car insurance provides peace of mind, ensuring that in the event of an unforeseen incident, the financial impact is minimized and the recovery process is as smooth as possible.
    """,

    "d5": """
    The car is distinct from the insurance policy, yet the two are inextricably linked in the world of automotive ownership and law. 
    The car is the physical asset, a machine of metal, rubber, and glass designed for transportation, subject to wear, tear, and mechanical failure. 
    The insurance policy, on the other hand, is a legal contract, an intangible promise of financial indemnification against specific risks associated with the car's operation. 
    Owning a car brings the joy of mobility but also the burden of risk; the policy is the tool used to manage that risk. 
    It is important to understand that the policy does not cover the car in all circumstances; for example, standard auto insurance typically does not cover mechanical breakdowns or routine maintenance, which are the responsibility of the owner. 
    The policy is defined by its terms, conditions, exclusions, and limits, which outline exactly what the insurer will and will not pay for. 
    While the car depreciates over time, the cost of the insurance policy may fluctuate based on the driver's behavior and external market factors. 
    Understanding this distinction is crucial for owners to realize that maintaining the vehicle is their job, while protecting the financial value of the vehicle against accidents is the job of the policy.
    """,

    "d6": """
    Python is a powerful programming language for data science, widely acclaimed for its simplicity, readability, and the vast ecosystem of libraries it supports. 
    Its syntax is designed to be intuitive and close to human language, which lowers the barrier to entry for beginners and allows experts to prototype rapid solutions. 
    In the realm of data science, Python acts as a glue language, seamlessly integrating various tools and workflows. 
    It dominates the field thanks to libraries like Pandas for data manipulation, NumPy for numerical computation, and Matplotlib or Seaborn for data visualization. 
    These tools allow data scientists to clean, analyze, and visualize massive datasets efficiently without writing low-level code. 
    Furthermore, Python's versatility extends beyond just analysis; it is robust enough to be used in production environments for web development and automation. 
    The community support for Python is immense, meaning that for almost any data-related problem, there is likely an existing library or a community forum discussion providing a solution. 
    This combination of ease of use, powerful libraries, and community support makes Python the de facto language for modern data science.
    """,

    "d7": """
    Machine learning and artificial intelligence are transforming technology at an unprecedented pace, reshaping industries and daily life. 
    This transformation is driven by the ability of AI systems to process vast amounts of data and identify patterns that were previously undetectable by human analysts. 
    In healthcare, AI is being used to diagnose diseases earlier and with greater accuracy, potentially saving countless lives. 
    In the automotive industry, machine learning algorithms are the brains behind self-driving cars, processing inputs from cameras and sensors to navigate complex traffic environments. 
    The financial sector utilizes these technologies for fraud detection and algorithmic trading, operating at speeds far beyond human capability. 
    Beyond these specific applications, AI is fundamentally changing human-computer interaction through natural language processing, enabling voice assistants and real-time translation services. 
    However, this technological revolution also brings challenges, such as ethical concerns regarding privacy, bias in algorithms, and the displacement of jobs. 
    Despite these challenges, the trajectory is clear: AI and machine learning are not just passing trends but foundational technologies that will define the future of innovation and infrastructure.
    """,

    "d8": """
    The best programming language depends on your specific needs, goals, and the constraints of the project you are undertaking. 
    There is no universal "silver bullet" in coding; each language was created to solve specific problems or to operate within certain environments. 
    For instance, if you are developing high-performance video games or system-level software where memory management and speed are critical, C++ might be the best choice due to its efficiency. 
    Conversely, if you are building a dynamic website, JavaScript is essential for front-end development, while Python or Ruby might be preferred for the back-end due to their developer-friendly syntax and rapid development cycles. 
    For data science and machine learning, Python is the undisputed leader, whereas R maintains a strong foothold in academic statistics. 
    Mobile app development might require Swift for iOS or Kotlin for Android. 
    Therefore, asking "what is the best language" is a misguided question; the better question is "what is the right tool for this job?" 
    Successful developers often learn multiple languages to have a diverse toolkit, allowing them to adapt to the specific requirements of any given technological challenge.
    """,

    "d9": """
    Data science combines statistics, programming, and domain knowledge to extract meaningful insights from structured and unstructured data. 
    It is an interdisciplinary field that sits at the intersection of quantitative analysis and computer science. 
    Statistics provides the mathematical foundation, offering the tools to understand probability, distributions, and significance, ensuring that conclusions drawn from data are valid and not just random noise. 
    Programming, typically in languages like Python or R, provides the mechanism to manipulate data, implement algorithms, and automate the analysis of datasets that are too large for manual processing. 
    However, technical skills alone are insufficient; domain knowledge is the secret sauce that allows a data scientist to ask the right questions and interpret the results in a relevant context. 
    Whether it is finance, healthcare, or marketing, understanding the specific industry allows the data scientist to distinguish between trivial correlations and actionable business insights. 
    This unique blend of skills makes data scientists highly sought after, as they bridge the gap between raw technical data and strategic decision-making.
    """,

    "d10": """
    Artificial intelligence systems can learn from experience, a capability that mimics the cognitive processes of humans and distinguishes AI from traditional static software. 
    This learning process is primarily achieved through machine learning algorithms, which improve their performance as they are exposed to more data over time. 
    Unlike traditional programming, where a developer explicitly codes every rule and outcome, AI systems are designed to infer rules from examples. 
    For instance, an image recognition system is not told exactly what a cat looks like in terms of pixels; instead, it is fed thousands of images of cats and non-cats, eventually learning to identify the defining features of a cat on its own. 
    This process involves training, where the model adjusts its internal parameters to minimize errors, and inference, where it applies what it has learned to new, unseen data. 
    Deep learning, a subset of machine learning using neural networks, has particularly excelled in this area, enabling systems to master complex tasks like playing Go or driving cars by continuously refining their internal models based on successes and failures.
    """,

    "d11": """
    Insurance companies offer various types of coverage options to allow policyholders to customize their protection based on their specific risks and budget. 
    A standard auto insurance policy is actually a package of different coverages. 
    Liability coverage is the foundation, paying for injuries or damage you cause to others. 
    Beyond that, collision coverage pays to repair your own car after an accident, regardless of who was at fault. 
    Comprehensive coverage steps in for non-collision incidents, such as theft, vandalism, fire, or hitting an animal. 
    Personal Injury Protection (PIP) or Medical Payments coverage helps pay for medical expenses for you and your passengers. 
    Uninsured/Underinsured Motorist coverage is crucial in protecting you if you are hit by a driver who lacks adequate insurance. 
    Additionally, insurers offer add-ons like rental reimbursement, towing and labor, and gap insurance. 
    Understanding these options is essential because opting for the bare minimum might save money on premiums but leaves the driver vulnerable to significant financial loss in a serious incident.
    """,

    "d12": """
    Python programming language is widely used in machine learning, having established itself as the lingua franca of the AI community. 
    Its dominance is not due to raw execution speed—languages like C++ are faster—but due to its developer efficiency and a massive ecosystem of specialized libraries. 
    Frameworks like TensorFlow, PyTorch, Keras, and Scikit-learn are all Python-based or have first-class Python bindings, making complex algorithms accessible via simple API calls. 
    This allows researchers and engineers to focus on the logic of their models rather than the intricacies of memory management or low-level implementation details. 
    Python's clean syntax facilitates rapid prototyping, which is essential in machine learning where experimentation and iteration are constant. 
    Furthermore, Python acts as a glue language, easily integrating with high-performance C/C++ code under the hood, giving developers the best of both worlds: the ease of scripting with the performance of compiled languages. 
    This accessibility has democratized machine learning, allowing students, researchers, and startups to build state-of-the-art AI applications.
    """,

    "d13": """
    The King of Pop revolutionized the music industry in ways that extended far beyond record sales. 
    Michael Jackson transformed the promotional landscape by elevating the music video from a simple marketing tool to a high-budget, narrative-driven art form. 
    Videos like 'Thriller', 'Beat It', and 'Black or White' became global events, premiering on primetime television and driving the popularity of the MTV network. 
    He shattered racial barriers in the music industry, becoming the first black artist to receive heavy rotation on MTV, which opened doors for countless artists of color who followed. 
    Sonically, his production standards, particularly his collaboration with Quincy Jones, set a new benchmark for audio engineering, blending analog and digital sounds to create pop music that was structurally complex yet incredibly catchy. 
    His marketing strategies, massive world tours, and merchandising deals created the blueprint for the modern pop superstar. 
    By combining image, sound, and dance into a singular, cohesive package, he completely rewrote the rules of engagement for the entertainment business.
    """,

    "d14": """
    Auto insurance rates vary based on driving history, a critical factor that insurers use to assess the risk of a potential policyholder. 
    Actuarial science dictates that past behavior is a strong predictor of future behavior; therefore, drivers with a history of accidents or traffic violations are statistically more likely to file claims. 
    A clean driving record is often rewarded with lower premiums and "safe driver" discounts. 
    Conversely, speeding tickets, DUIs, or at-fault accidents signal high risk, leading to significantly increased rates or even policy cancellation. 
    However, driving history is not the only variable; insurers also consider age, gender, location, credit score, and vehicle type. 
    Younger drivers, particularly teenagers, face the highest rates due to their lack of experience. 
    Living in a densely populated urban area with high theft rates will also drive up premiums compared to rural areas. 
    Ultimately, the insurance rate is a calculated risk assessment, with the driving history serving as one of the most heavily weighted components in the algorithmic pricing model.
    """,

    "d15": """
    Machine learning algorithms require large amounts of data to function effectively, a dependency that has coined the phrase "data is the new oil." 
    These algorithms, particularly deep learning neural networks, contain millions of parameters that must be tuned during the training process. 
    To accurately tune these parameters and avoid overfitting—where the model memorizes the training data rather than learning generalizable patterns—vast datasets are necessary. 
    The quality of this data is just as important as the quantity; noisy, biased, or incorrectly labeled data can lead to poor model performance or unethical outcomes. 
    This requirement has given rise to the Big Data era, where companies aggressively collect user data to fuel their AI systems. 
    From image recognition systems trained on millions of photos to language models trained on the entire text of the internet, the capability of modern AI is directly proportional to the scale of data available. 
    Consequently, data preprocessing, cleaning, and augmentation have become critical steps in the machine learning pipeline.
    """,

    "d16": """
    Programming in Python is easier for beginners to learn compared to many other languages, making it the most recommended entry point for computer science education. 
    Its syntax was designed with readability in mind, often resembling the English language, which reduces the cognitive load on new students who are struggling to understand programming concepts. 
    Unlike languages like Java or C++, Python handles much of the complexity, such as memory management and variable declaration, automatically. 
    This allows beginners to focus on problem-solving and algorithmic thinking rather than fighting with syntax errors or compilation issues. 
    Python is an interpreted language, meaning code can be run line-by-line, providing immediate feedback which is crucial for the learning process. 
    Furthermore, the wealth of free educational resources, tutorials, and a supportive community makes it incredibly accessible. 
    Whether a beginner wants to build a simple game, a website, or analyze data, Python provides quick wins that encourage continued learning and development.
    """,

    "d17": """
    The insurance policy covers damages from accidents, providing a financial shield against the unpredictable nature of driving. 
    When an accident occurs, the policyholder files a claim, initiating a process where the insurer assesses the damage and liability. 
    If the policyholder is at fault, their liability coverage pays for the other party's medical bills and vehicle repairs, protecting the policyholder's personal assets from lawsuits. 
    If the policyholder has collision coverage, the insurer will also pay to repair the policyholder's vehicle, minus the deductible. 
    It is important to note that coverage limits apply; if the damages exceed the limit specified in the policy, the driver is responsible for the difference. 
    Additionally, policies have exclusions—for example, they might not cover accidents that occur while using the vehicle for commercial purposes like ride-sharing, unless a specific endorsement is added. 
    Understanding exactly what accidents are covered and to what extent is vital for ensuring that a minor fender bender doesn't turn into a major financial crisis.
    """,

    "d18": """
    Michael Jackson's legacy continues to inspire musicians across virtually every genre of music, from pop and R&B to hip-hop and rock. 
    His influence is audible in the vocal stylings of artists like The Weeknd, Bruno Mars, and Justin Timberlake, who have all cited him as a primary inspiration. 
    Beyond vocals, his impact on stage presence and dance is undeniable; the precision, fluidity, and energy he brought to live performance set a standard that modern pop stars strive to emulate. 
    He pioneered the concept of the "triple threat"—an artist who could sing, dance, and write their own material at an elite level. 
    His fashion sense remains influential, with elements of his iconic looks appearing in high fashion and streetwear. 
    Furthermore, his humanitarian efforts and messages of unity in songs like 'Man in the Mirror' and 'Heal the World' continue to resonate. 
    Despite the controversies that surrounded his life, his artistic contributions remain a cornerstone of modern music history, proving that his creative DNA is deeply woven into the fabric of popular culture.
    """,

    "d19": """
    Data science projects require both technical and analytical skills, creating a demand for professionals who are versatile "unicorns" in the job market. 
    On the technical side, proficiency in programming languages like Python or R, and SQL for database management, is non-negotiable. 
    Data scientists must be comfortable manipulating large datasets, building machine learning models, and using version control systems like Git. 
    However, coding ability is useless without strong analytical skills—the mathematical intuition to select the right statistical tests and the critical thinking to interpret results correctly. 
    Moreover, soft skills are increasingly important; a data scientist must be a storyteller, capable of translating complex algorithmic outputs into clear, actionable business insights for non-technical stakeholders. 
    They must understand the business domain to solve the right problems and avoid getting lost in theoretical optimization. 
    Successful projects rely on this synthesis of hard engineering skills, rigorous mathematical analysis, and clear communication.
    """,

    "d20": """
    Artificial intelligence is reshaping how we interact with technology, moving us away from command-based interfaces towards more natural, conversational interactions. 
    Voice assistants like Siri, Alexa, and Google Assistant have made it possible to control our environments, access information, and communicate using spoken language, lowering the barrier to technology adoption for children and the elderly. 
    Recommendation algorithms on platforms like Netflix, Spotify, and Amazon subtly guide our choices, personalizing our digital experiences based on behavioral data. 
    In the workspace, AI-powered tools assist with writing, scheduling, and data analysis, acting as intelligent collaborators rather than just passive tools. 
    We are seeing the rise of ambient computing, where technology fades into the background, proactively anticipating our needs through predictive AI. 
    This shift fundamentally changes the relationship between human and machine; we no longer just operate computers, we partner with them. 
    As these technologies advance, the line between tool and assistant blurs, creating a future where technology is woven seamlessly into the fabric of daily life.
    """,
}

# 2. Pretraitement (Tokenisation + Normalisation + Stopwords)
# Notes:
# - Tokenisation regex pour enlever ponctuation.
# - Stopwords minimalistes (anglais) pour un mini-corpus.
STOPWORDS: Set[str] = set([
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "the", "this", "but", "they",
    "have", "had", "what", "when", "where", "who", "which", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "just", "should", "now", "one", "two", "three"
])

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    """Tokenise un texte (ponctuation ignoree)."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def preprocess(text: str, *, stopwords: Set[str] = STOPWORDS) -> List[str]:
    """Tokenisation + normalisation + filtrage stopwords."""
    tokens = tokenize(text)
    if not tokens:
        return []
    return [t for t in tokens if t not in stopwords]


def build_corpus_processed(docs: Dict[str, str]) -> Dict[str, List[str]]:
    return {doc_id: preprocess(text) for doc_id, text in docs.items()}


def build_vocabulary(corpus: Mapping[str, Iterable[str]]) -> Set[str]:
    return set(word for tokens in corpus.values() for word in tokens)

# Exports historiques (compatibilite avec le reste du projet)
corpus_processed = build_corpus_processed(documents)
vocabulary = build_vocabulary(corpus_processed)