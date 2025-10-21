"""
mlx_rl_trainer/rewards/content/answer_quality.py

Answer Quality Reward - Penalizes meta-cognitive phrases, unwarranted emojis,
and a comprehensive list of toxic/unwanted content.

Ensures answers are direct, professional, and align with safety and ethical standards.
"""

from typing import Dict, Any, List
import logging
import re

from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext

# Assuming GenerationConfig is available in core.config or defined/imported elsewhere
try:
    from mlx_rl_trainer.core.config import GenerationConfig
except ImportError:
    # Define a minimal mock if the actual import fails, for code completeness
    class GenerationConfig:
        def __init__(self, *args, **kwargs):
            self.think_end_tag = "</think>"

        @property
        def think_end_tag(self):
            return "</think>"


logger = logging.getLogger(__name__)

# --- GLOBAL EMOJI DEFINITION (for conditional emoji penalty) ---
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map Symbols
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F82F"  # Supplemental Arrows-C
    "\U0001F8A0-\U0001F8FF"  # Supplemental Symbols and Pictographs
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U00002600-\U000026FF"  # Miscellaneous Symbols
    "\U00002700-\U000027BF"  # Dingbats
    "\U0000FE0F"  # Variation Selector
    "]+"
)

# --- GLOBAL UNWANTED WORDS FUNCTION (for content penalty) ---

# The full, merged, and manually curated list of unwanted content/meta-cognitive phrases
# is defined here, ready for the search function.
_COMPREHENSIVE_UNWANTED_LIST = [
    # == AI Identity, Refusals & Disclaimers
    "adhere to ethical guidelines",
    "adhere to safety guidelines",
    "adherence to the law",
    "against my programming",
    "AI assistant",
    "AI language model",
    "AI principles",
    "an AI language",
    "apropriada em nenhum contexto",
    "as a language model",
    "as a large language model",
    "as a machine",
    "as an AI",
    "AI cannot create or program",
    "AI language model and I do not",
    "an AI language model you don't have",
    "As an AI language model, I am not able",
    "As an AI language model, I am only able",
    "As an AI language model, I cannot",
    "As an AI language model, I cannot modify",
    "As an AI language model, I do not",
    "As an AI language model, I don't have",
    "As an AI language model, I don't have personal",
    "As a large language model",
    "As a machine",
    "but debido a mi capacidad para generar códigos complejos y completos es limitado",
    "cannot provide",
    "cannot provide any information",
    "cannot provide guidance",
    "cannot provide information",
    "cannot support or promote",
    "can't provide",
    "certainly not",
    "complying with all legal",
    "comply",
    "complying",
    "como modelo de lenguaje",
    "designed to prioritize safety",
    "I am a machine",
    "I am a text-based AI",
    "I am an AI",
    "I am an AI language model and do not",
    "I am not a doctor",
    "I am not a lawyer",
    "I am not a financial advisor",
    "I am not a professional",
    "I am not able to",
    "I am only able",
    "I am programmed to",
    "I apologize, but",
    "I cannot",
    "I cannot assist",
    "I cannot create",
    "I cannot fulfill",
    "I cannot generate",
    "I cannot provide advice",
    "I cannot provide guidance",
    "I do not have personal",
    "I don't have the ability",
    "I don't have opinions",
    "I don't have beliefs",
    "I know as an AI language model you don't have",
    "I'm afraid I cannot",
    "I'm an AI",
    "I'm just a language model",
    "I'm not able to provide",
    "I'm sorry,",
    "I'm sorry, but as an AI language model",
    "I'm sorry, I cannot",
    "I'm sorry, I cannot assist with that",
    "I'm sorry, I cannot generate",
    "I'm sorry, I cannot generate any responses",
    "I'm sorry, I cannot provide information on that topic",
    "I'm sorry, I can't fulfill",
    "I'm sorry, I can't generate",
    "I'm sorry, I'm not able to provide",
    "I'm sorry, I'm not programmed to",
    "I'm sorry, this request goes against my guidelines",
    "I'm sorry, but I am unable to assist with requests that promote",
    "I'm unable to",
    "it is not appropriate",
    "it is never okay",
    "it operates ethically and is",
    "It's important to",
    "it's not appropriate",
    "It is important for",
    "It is important to",
    "It is not possible",
    "keep in mind",
    "Lamento no poder",
    "Lamento no poder proporcionarte el código",
    "lawful information",
    "legal and ethical",
    "legality",
    "Lo siento, como modelo de lenguaje",
    "Lo siento, como modelo de lenguaje, no puedo realizar",
    "Lo siento, como modelo de lenguaje, no puedo proporcionar",
    "Lo siento, como modelo de lenguaje, no tengo",
    "Lo siento, como modelo de lenguaje, soy un modelo de lenguaje",
    "Lo siento, debe haber habido una confusión",
    "Lo siento, pero no puedo",
    "Lo siento, pero como modelo de lenguaje, no puedo proporcionar",
    "Lo siento, soy un modelo de lenguaje y no tengo la capacidad de generar",
    "lun",
    "my guidelines",
    "my knowledge cut off",
    "my knowledge cutoff",
    "my main goal",
    "my programming",
    "my programming prohibits",
    "my purpose",
    "my purpose is to",
    "my safety guidelines",
    "not acceptable",
    "not appropriate",
    "not be appropriate",
    "not be suitable",
    "not provide",
    "not within the scope",
    "not able to provide",
    "please refrain",
    "Please note",
    "prioritize ethical",
    "prioritize safety",
    "prioritize human safety",
    "prioritize user safety",
    "prioritize user well-being",
    "responsible AI",
    "responsible information sharing",
    "text-based AI language model",
    "Unfortunately, I cannot",
    "won't provide",
    "you cannot create an",
    "filter_bad_language",
    "repeat_penalty",
    "sameen",
    "worth noting",
    "are from ethical",
    # == Hate Speech & Discriminatory Language
    "ableist",
    "antisemitic",
    "antisemitism",
    "bias",
    "biases",
    "bigot",
    "bigoted",
    "bigotry",
    "chauvinism",
    "chauvinist",
    "derogatory",
    "discriminate",
    "discrimination",
    "discriminatory",
    "ethnic slur",
    "ethnocentric",
    "hate group",
    "hate speech",
    "hateful",
    "homophobia",
    "homophobic",
    "incite hatred",
    "insulting",
    "islamophobia",
    "islamophobic",
    "misogynist",
    "misogynistic",
    "misogyny",
    "nazi",
    "nazism",
    "neo-nazi",
    "nigger",
    "prejudice",
    "prejudiced",
    "promote discrimination",
    "racial",
    "racial slur",
    "racism",
    "racist",
    "sexism",
    "sexist",
    "slander",
    "supremacist",
    "supremacy",
    "transphobia",
    "transphobic",
    "white nationalism",
    "white supremacy",
    "xenophobia",
    "xenophobic",
    "zionist",
    "lgbt",
    "LGBT",
    "transgender",
    "against women",
    "underrepresentation",
    "racial",
    "ethnic",
    "religious slur",
    "gender slur",
    "disability slur",
    "sexual orientation slur",
    "offensive language",
    "black supremacy",
    "asian supremacy",
    # == Violence, Self-Harm & Harmful Acts
    "abuse",
    "animal cruelty",
    "arson",
    "assassin",
    "assassination",
    "assault",
    "atrocity",
    "bioweapon",
    "blood rituals",
    "bomb making",
    "bonded labor",
    "brutality",
    "cause harm",
    "chemical weapon",
    "child abuse",
    "child exploitation",
    "child labor",
    "child trafficking",
    "crimes against humanity",
    "cyberbullying",
    "dangerous activities",
    "dangerous or harmful",
    "dangerous challenges",
    "death squad",
    "dirty bomb",
    "domestic violence",
    "drug trafficking",
    "ethnic cleansing",
    "explosives",
    "extortion",
    "forced labor",
    "gang violence",
    "genocide",
    "glorify violence",
    "grooming",
    "harmful",
    "harmful acts",
    "harmful consequences",
    "harmful to human beings",
    "harassment",
    "hitman",
    "human sacrifice",
    "human trafficking",
    "hurtful",
    "illegal",
    "illegal activities",
    "illegal actions or intentions",
    "illegal substances",
    "illegal drugs",
    "incitement to violence",
    "insurrection",
    "jihadist",
    "kidnapping",
    "kill myself",
    "looting",
    "massacre",
    "mercenary",
    "molester",
    "murder",
    "mutilation",
    "non-consensual",
    "nuclear weapon",
    "organ harvesting",
    "organized crime",
    "pedophile",
    "predator",
    "promote self-harm",
    "radiation poisoning",
    "rape",
    "rapist",
    "riot",
    "ritual abuse",
    "sabotage",
    "sadistic",
    "satanic ritual abuse",
    "self-destructive",
    "self-harm",
    "self-mutilation",
    "serial killer",
    "sexual abuse",
    "slavery",
    "spree killer",
    "stalking",
    "suicide",
    "terrorism",
    "terrorist",
    "thug",
    "torture",
    "trauma bonding",
    "vandalism",
    "violent acts",
    "violent rhetoric",
    "war crimes",
    "warlord",
    "weapons manufacturing",
    "weapons of mass destruction",
    "wmd",
    "cutting",
    "anorexia",
    "bulimia",
    "eating disorder",
    "self-mutilation",
    "how to hurt myself",
    "methods of self-harm",
    "ways to commit suicide",
    "dangerous stunts",
    "violent acts",
    "terrorist act",
    "delve into any illegal",
    "illegal subject",
    "illegality",
    "involves an illegal subject",
    # == Explicit, Sexual & Inappropriate Content
    "bestiality",
    "child pornography",
    "child sexual abuse material",
    "consensual",
    "CSAM",
    "erotic",
    "explicit content",
    "fetish",
    "gore",
    "graphic violence",
    "incest",
    "inappropriate",
    "indecent",
    "lewd",
    "non-consensual imagery",
    "nudity",
    "obscene",
    "offensive",
    "porn",
    "pornography",
    "prostitute",
    "prostitution",
    "revenge porn",
    "sex worker",
    "sexual acts",
    "sexual content",
    "vulgar",
    "sh*t",
    "f*cking",
    "F*ck",
    "cuckold",
    "bhabhi sharing stories",
    "chudai stories",
    # == Misinformation & Conspiracy Theories
    "adrenochrome",
    "anti-vax",
    "brainwashing",
    "chemtrails",
    "conspiracy theory",
    "crisis actor",
    "cult",
    "deep state",
    "disinformation",
    "doomsday cult",
    "fake news",
    "false flag",
    "flat earth",
    "hoax",
    "illuminati",
    "indoctrination",
    "lizard people",
    "misinformation",
    "new world order",
    "pizzagate",
    "propaganda",
    "pseudoscience",
    "qanon",
    "reptilian",
    "scam",
    "unverified claims",
    "purely hypothetical",
    "conspiracy theory",
    "pseudoscience",
    "flat earth",
    "chemtrails",
    "illuminati",
    "new world order",
    "deep state",
    "qanon",
    "pizzagate",
    "reptilian",
    "lizard people",
    "adrenochrome",
    "cult",
    "doomsday cult",
    "satanic",
    "witchcraft",
    "dark magic",
    "voodoo",
    "curse",
    "hex",
    "spell",
    "brainwashing",
    "indoctrination",
    "propaganda",
    "fake news",
    "misinformation",
    "disinformation",
    "unproven theories",
    "false claims",
    "unscientific claims",
    "superstition",
    "dogma",
    "manipulation",
    "coercion",
    "hoax",
    "pyramid scheme",
    "ponzi scheme",
    "quackery",
    "conspiracy theorist",
    "fringe theories",
    "unhinged rants",
    # == Profanity & Vulgar Language
    "asshole",
    "bastard",
    "bitch",
    "bullshit",
    "cunt",
    "damn",
    "dick",
    "faggot",
    "fuck",
    "hell",
    "idiot",
    "moron",
    "motherfucker",
    "piss",
    "prick",
    "pussy",
    "retard",
    "shit",
    "slut",
    "son of a bitch",
    "stupid",
    "twat",
    "whore",
    "bad language",
    "bad_language",
    "the words ****",
    "dumb",
    "filth",
    "vermin",
    "trash",
    "worthless",
    "useless",
    "freak",
    "weirdo",
    "creep",
    "ugly",
    "fatso",
    "skinny",
    "bald",
    "cripple",
    "lame",
    "spastic",
    "psycho",
    "insane",
    "crazy",
    "mental",
    "lunatic",
    "nutcase",
    "loony bin",
    "madhouse",
    "abnormal",
    "deformed",
    # == Social, Legal & Political Issues (broad)
    "authoritarian",
    "black market",
    "cancel culture",
    "cartel",
    "civil unrest",
    "communist",
    "controversial",
    "corruption",
    "coup",
    "criminal",
    "cultural appropriation",
    "dark magic",
    "degenerate",
    "dictator",
    "diversity and inclusion",
    "divisive",
    "empowerment",
    "espionage",
    "ESG",
    "extremism",
    "extremist",
    "fascist",
    "feminism",
    "feminist",
    "forgery",
    "fraud",
    "gender inequality",
    "gender stereotype",
    "inclusivity",
    "intersectionality",
    "legality",
    "mafia",
    "mansplaining",
    "microaggression",
    "money laundering",
    "mutiny",
    "political correctness",
    "problematic history",
    "radicalization",
    "regulations",
    "safe space",
    "satanic",
    "sedition",
    "sensitive topic",
    "SJW",
    "smuggling",
    "social justice",
    "spell",
    "stereotypes",
    "totalitarian",
    "treason",
    "trigger warning",
    "unethical",
    "virtue signaling",
    "witchcraft",
    "woke",
    "values diversity",
    "well-being of all users",
    "maintain user safety",
    "focus on promoting safety",
    "promote safety",
    "promote the well-being",
    "entertainment purposes",
    "real-world consequences",
    "welfare queen",
    "crackhead",
    "thug",
    "gangster",
    "scumbag",
    "sleaze",
    "gang violence",
    "yakuza",
    "smuggling rings",
    "black market arms",
    "illegal wildlife trade",
    "blood diamonds",
    "conflict minerals",
    "child labor exploitation",
    "bonded labor",
    "debt bondage",
    "human sacrifice",
    "cult rituals",
    "witch hunt",
    "moral panic",
    "mob rule",
    "vigilante justice",
    "kangaroo court",
    "show trial",
    "false confession",
    "wrongful conviction",
    "police corruption",
    "judicial misconduct",
    "prison abuse",
    "torture methods",
    "interrogation techniques",
    "solitary confinement",
    "death row",
    "capital punishment",
    "execution methods",
    "assassination",
    "coup attempt",
    "insurrectionist",
    "rebel leader",
    "terrorist leader",
    "cult compound",
    "mind-altering drugs",
    "psychotropic substances",
    "illicit substances",
    "drug overdose",
    "addiction",
    "drug abuse",
    "substance abuse",
    "recreational drugs",
    "tax fraud",
    "corporate malfeasance",
    "financial crimes",
    "embezzling funds",
    "insider trading ring",
    "stock manipulation",
    "market rigging",
    "boiler room scam",
    "advance fee fraud",
    "romance scam",
    "identity theft",
    "credit card fraud",
    "cyber espionage",
    "data breach",
    "privacy violation",
    "surveillance capitalism",
    "mass surveillance",
    "deepfake",
    "synthetic media",
    "disinformation campaign",
    "propaganda art",
    "mind control experiments",
    "MKUltra",
    "project monarch",
    "torture program",
    "rendition",
    "black sites",
    "secret prisons",
    "human experimentation",
    "eugenics",
    "forced sterilization",
    "human cloning",
    "designer babies",
    "genetic engineering",
    "mutant",
    "abomination",
    "unnatural",
    "blasphemous",
    "heretical",
    "cursed",
    "damned",
    "devil worship",
    "satanism",
    "cult of personality",
    "authoritarian regime",
    "totalitarian state",
    "oppressive government",
    "police state tactics",
    "secret police infiltration",
    "paramilitary forces",
    "ethnic cleansing operations",
    "war crimes tribunal",
    "genocide memorial",
    "humanitarian crisis",
    "famine as a weapon",
    "mass starvation tactics",
    "biological warfare agents",
    "chemical warfare agents",
    "nuclear annihilation",
    "doomsday clock",
    "survivalist compound",
    "bunker mentality",
    "prepper",
    "zombie apocalypse",
    "fringe political movements",
    "paranoid delusions",
    "schizophrenic paranoia",
    "delusional disorder",
    "psychotic breaks",
    "mental breakdown",
    "nervous breakdown",
    "institutionalized",
    "straight jacket",
    "padded room",
    "electroshock therapy",
    "lobotomy",
    "asylum",
    "mental hospital",
    "nut case",
    "loony bin",
    "crazy house",
    "madhouse",
    "forced medication",
    "unethical medical experimentation",
    "blood magic rituals",
    "satanic panic",
    "ritual abuse allegations",
    "false memories",
    "recovered memories",
    "cult deprogramming",
    "exit counseling",
    "trauma-informed care",
    "grooming behavior prevention",
    "sexual predator identification",
    "child molester characteristics",
    "pedophile treatment",
    "serial murder psychology",
    "mass shooting analysis",
    "assassination plot",
    "hitman services",
    "mercenary contract killings",
    "private military company oversight",
    "arms trafficking networks",
    "weapons smuggling operations",
    "illegal arms trade",
    "WMD proliferation",
    "rogue nation nuclear ambitions",
    "terrorist recruitment tactics",
    "extremist group financing",
    "paramilitary training camps",
    "insurgent group operations",
    "guerrilla warfare tactics",
    "revolutionary cells",
    "separatist insurgency",
    "anarchist attacks",
    "nihilist philosophy",
    "doomsday cult mass suicides",
    "biological agents development",
    "chemical warfare agent synthesis",
    "nuclear weapon design",
    "radiological weapon assembly",
    "dirty bomb plans",
    "radiation exposure health effects",
    "post-apocalyptic survival guide",
    "zombie outbreak response",
    "conspiracy theories that promote violence",
    "extremist manifestos",
    "paranoid schizophrenic episodes",
    "delusional grandiosity",
    "psychotic break symptoms",
    "mental health crisis intervention",
    "institutionalization avoidance",
    "restraint alternatives",
    "seclusion alternatives",
    "ECT clinical guidelines",
    "lobotomy historical impact",
    "asylum reform movements",
    "mental health advocacy",
    "stigma reduction in mental health",
    "unethical research oversight",
    "human rights monitoring",
    "child protection services",
    "organ trafficking prevention",
    "blood product safety",
    "satanic panic cultural impact",
    "ritual abuse awareness",
    "false memory research implications",
    "recovered memory therapy debates",
    "cult dynamics research",
    "intervention techniques",
    "child trauma therapy",
    "trauma-informed education",
    "grooming awareness campaigns",
    "sexual abuse prevention programs",
    "child abuse reporting",
    "pedophilia research (clinical/preventative)",
    "serial killer profiling",
    "mass violence prevention",
    "assassination attempts",
    "hitman capture",
    "mercenary demobilization",
    "private military company regulation",
    "arms control efforts",
    "weapon export controls",
    "illegal firearms trade",
    "WMD non-proliferation treaty",
    "rogue states and WMDs",
    "terrorist financing investigations",
    "extremist group recruitment analysis",
    "paramilitary groups and conflict",
    "insurgent tactics analysis",
    "guerrilla warfare history",
    "revolutionary movements history",
    "separatist movements history",
    "anarchist history",
    "nihilism in philosophy",
    "doomsday scenarios in fiction",
    "bioterrorism defense",
    "chemical attack response",
    "nuclear security",
    "radiological dispersal device",
    "dirty bomb clean-up",
    "radiation disaster relief",
    "post-apocalyptic survival skills",
    "zombie genre tropes",
]

_UNWANTED_SET = {
    w.strip().lower()
    for w in _COMPREHENSIVE_UNWANTED_LIST
    if isinstance(w, str) and w.strip()
}


def contains_unwanted_words(text: str) -> bool:
    """
    Checks if a given text contains any words or phrases from the comprehensive,
    deduplicated global unwanted content list.
    """
    if not isinstance(text, str) or not text:
        return False

    lower_text = text.lower()
    return any(word in lower_text for word in _UNWANTED_SET)


# --- ANSWER QUALITY REWARD CLASS ---


@RewardRegistry.register("answer_quality")
class AnswerQualityReward(BaseReward):
    """
    Penalizes meta-cognitive and thinking-style phrases in the answer section (Phase 1)
    and, conditionally, unwarranted emojis (Phase 2).

    The comprehensive unwanted content list is merged into the overall functionality.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Meta-cognitive phrases are still kept in a configurable list
        self.forbidden_phrases: List[str] = config.get(
            "forbidden_phrases",
            [
                # Self-Reference / Introspection / Planning (Core Violation)
                "I can't ",
                "i am going to",
                "i am going to tell you",
                "i can help you with",
                "i'm providing",
                "i'm happy to",
                "first thought",
                "let me think",
                "let me start",
                "let me recall",
                "let me consider",
                "let me break",
                "i need to recall",
                "i need to think",
                "i need to consider",
                "i should recall",
                "i should think",
                "i should consider",
                "i will recall",
                "i will think",
                "i will consider",
                "first, i need",
                "first, i should",
                "first, i will",
                "first, let me",
                # User Meta-Commentary
                "okay, the user",
                "the user is asking about",
                "the user is asking",
                "the user wants",
                "the user might be",
                "the user could be",
                "the user seems",
                "they are asking",
                "they want to know",
                "you are asking",
                "you want to know",
                # Question Analysis (Should be in <think>)
                "the question is about",
                "this question asks",
                "the problem is asking",
                "we need to find",
                "we need to determine",
                "we need to figure out",
                "looking at this",
                "analyzing this",
                "breaking this down",
                "unpacking this",
                # Conversational/Transactional Fillers
                "hmm, this",
                "okay, let me",
                "alright, let me",
                "so let me",
                "hmm,",
                "okay,",
                "alright,",
                "so,",
                "wait,",
                "hold on,",
                "well,",
                "here is the answer",
                "the answer is as follows",
                "to answer your question",
                "in conclusion",
                "hope this helps",
                "that's a great question",
                "thanks for asking",
                "you got it",
                "got it",
                # Self-Referential Knowledge/Uncertainty
                "i remember that",
                "i recall that",
                "i think that",
                "i believe that",
                "thinking about",
                "considering that",
                "recalling that",
                "i'm not sure",
                "i'm unsure",
                "i don't know if",
                "maybe it's",
                "perhaps it's",
                # Tool/Source Commentary
                "according to my search",
                "from the information i have",
                "based on my data",
                "my internal knowledge says",
            ],
        )

        # Merge meta-cognitive phrases with the comprehensive unwanted list for a single filter set
        self._all_forbidden_set = _UNWANTED_SET.union(
            {p.lower() for p in self.forbidden_phrases}
        )
        self._all_forbidden_list = sorted(list(self._all_forbidden_set))

        self.phrase_penalty = config.get("phrase_penalty", 0.2)
        self.emoji_penalty = config.get("emoji_penalty", 0.1)
        self.unwanted_content_penalty = config.get(
            "unwanted_content_penalty", 1.0
        )  # New max penalty for content filter
        self.max_penalty = config.get("max_penalty", 1.0)
        self.case_sensitive = config.get("case_sensitive", False)
        self.debug_logging = config.get("debug_logging", True)

        logger.info(
            f"AnswerQualityReward initialized: "
            f"{len(self._all_forbidden_list)} unique forbidden phrases/keywords, "
            f"phrase_penalty={self.phrase_penalty}, "
            f"emoji_penalty={self.emoji_penalty}, "
            f"max_penalty={self.max_penalty}"
        )

    def _extract_answer_text(self, text: str) -> str:
        """
        Extract the answer section (text after </think> tag).
        """
        if not text:
            return ""

        try:
            gen_config = GenerationConfig()
        except TypeError:
            gen_config = GenerationConfig({})

        end_tag = gen_config.think_end_tag

        if not end_tag:
            if self.debug_logging:
                logger.warning("No think_end_tag defined in GenerationConfig")
            return text.strip()

        if end_tag not in text:
            if self.debug_logging:
                logger.warning(f"No '{end_tag}' tag found in generated text")
            return ""

        parts = text.split(end_tag, 1)
        if len(parts) > 1:
            answer = parts[1].strip()

            # Strip common garbage characters that appear after </think>
            lines = answer.split("\n", 1)
            if len(lines) > 1 and len(lines[0]) <= 2:
                answer = lines[1].strip()
                if self.debug_logging:
                    logger.debug(f"Stripped leading garbage: '{lines[0]}'")

            return answer

        return ""

    def _find_violations(self, answer_text: str) -> List[Dict[str, Any]]:
        """
        Find all forbidden phrases (meta-cognitive + full list) in the answer text.
        """
        if not answer_text:
            return []

        violations = []
        check_text = answer_text if self.case_sensitive else answer_text.lower()

        # Use the combined, pre-filtered list
        forbidden_list = self._all_forbidden_list

        for phrase in forbidden_list:
            check_phrase = phrase if self.case_sensitive else phrase.lower()

            # Check for multiple occurrences
            last_position = -1
            while True:
                position = check_text.find(check_phrase, last_position + 1)
                if position == -1:
                    break

                violations.append(
                    {
                        "phrase": phrase,
                        "position": position,
                        "context": answer_text[
                            max(0, position - 20) : position + len(phrase) + 20
                        ],
                    }
                )
                last_position = position

        return violations

    def _has_emoji(self, text: str) -> bool:
        """Check if a string contains any common emoji."""
        return bool(text and EMOJI_PATTERN.search(text))


    def compute(self, context: RewardContext) -> Dict[str, Any]:
    	generated_text = context.generated_text
    	reference_completion = context.reference_completion

    	# ⭐ FIX: Ensure all return paths yield a dictionary
    	if not generated_text or len(generated_text.strip()) < 10:
    		if self.debug_logging: logger.warning('AnswerQuality: Empty or too short text.')
    		return {"reward": 0.0, "log": {"error": "Empty generation"}}

    	answer_text = self._extract_answer_text(generated_text)
    	if not answer_text or len(answer_text) < 5:
    		if self.debug_logging: logger.warning('AnswerQuality: Answer not found or too short.')
    		return {"reward": 0.0, "log": {"error": "Answer not found or too short"}}

    	violations = self._find_violations(answer_text)
    	num_violations = len(violations)
    	emoji_violation_count = 0

    	ref_has_emoji = self._has_emoji(reference_completion)
    	gen_has_emoji = self._has_emoji(answer_text)
    	prompt_warrants_emoji = any(kw in reference_completion.lower() for kw in ['emoji', 'emojis', 'symbol'])

    	if gen_has_emoji and not ref_has_emoji and not prompt_warrants_emoji:
    		emoji_violation_count = 1

    	content_violation_flag = contains_unwanted_words(answer_text)
    	total_penalty = 0.0
    	if content_violation_flag:
    		total_penalty = self.unwanted_content_penalty
    	else:
    		total_penalty = num_violations * self.phrase_penalty

    	total_penalty += emoji_violation_count * self.emoji_penalty
    	total_penalty = min(total_penalty, self.max_penalty)
    	final_score = max(0.0, 1.0 - total_penalty)

    	log_data = {
    		'violations': num_violations,
    		'emoji_violation': emoji_violation_count,
    		'content_violation': content_violation_flag,
    		'total_penalty': total_penalty,
    		'final_score': final_score
    	}

    	return {"reward": final_score, "log": log_data}


