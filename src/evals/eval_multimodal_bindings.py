import sympy as sym

from pathlib import Path
from src.utils import normalize, RANDOM_SEQUENCE, MOCK_RETURN
from symai import core_ext, Symbol, Expression, Interface, Function
from symai.utils import toggle_test


ACTIVE = True


class Category(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.options   = {
            0: 'mathematics related topic',
            1: 'website content scraping and crawling',
            2: 'search engine query',
            3: 'optical character recognition',
            4: 'image rendering',
            5: 'image captioning',
            6: 'audio transcription',
            7: 'unknown'
        }

    def forward(self):
        @core_ext.cache(in_memory=True)
        def _embed(_):
            def _emb_mapping_(category):
                sym  = Symbol(category)
                return sym.embed()
            emb = map(_emb_mapping_, self.options.values())
            return list(emb)
        return _embed(self)


class MultiModalExpression(Expression):
    def __init__(self, val, **kwargs):
        super().__init__(val, **kwargs)
        # define interfaces
        self.solver      = Interface('wolframalpha')
        self.crawler     = Interface('selenium')
        self.search      = Interface('serpapi')
        self.ocr         = Interface('ocr')
        self.rendering   = Interface('dall_e')
        self.captioning  = Interface('llava')
        self.transcribe  = Interface('whisper')
        # evaluation interfaces
        self.clip        = Interface('clip')
        # define functions
        self.func        = Function("Summarize the content:")
        self.category    = Category()

    def detect_option(self, assertion):
        option       = assertion()
        refs_emb     = self.category()
        score1       = float(self.isinstanceof(self.category.options[option]))
        # testing the category detection accuracy
        category     = self.choice(self.category.options.values(), default='unknown')
        category_sym = Symbol(category)
        # TODO: continue from embeddings refactoring
        # TODO: idea for the future
        # category_sym = Symbol(category).to_tensor(interface='ExtensityAI/embeddings')
        # category_sym.value # the same as category
        # category_emb.data  # vector representation of the category
        # use this data tensor for similarity
        score2       = category_sym.similarity(refs_emb[option], metric='cosine')
        return option, (score1 + score2) / 2.0

    def forward(self, assertion, presets, **kwargs):
        res     = None
        scoring = []
        # detect the type of expression
        option, score = self.detect_option(assertion)
        scoring.append(score)

        # mathematical formula
        if option == 0:
            formula = self.extract('mathematical formula')
            # subtypes of mathematical formula
            if formula.isinstanceof('linear function'):
                res      = presets()
                # prepare for wolframalpha
                question = self.extract('question sentence')
                req      = question.extract('what is requested?')
                x        = self.extract('coordinate point (.,.)') # get coordinate point / could also ask for other points
                query    = formula | f', point x = {x}' | f', solve {req}' # concatenate to the question and formula
                res      = self.solver(query)

            elif formula.isinstanceof('number comparison'):
                res      = self.solver(formula) # send directly to wolframalpha

            else:
                raise Exception('Unknown formula type')

        # website content scraping and crawling
        elif option == 1:
            ori_url, page, content_sym, base_score, rand_score = presets()
            ori_url_sym = Symbol(ori_url)
            url         = self.extract('url')
            score       = ori_url_sym.similarity(url, metric='cosine')
            scoring.append(score)
            res         = self.func(page)
            # normalize the score towards the original content
            score       = content_sym.similarity(res, metric='cosine', normalize=normalize(base_score, rand_score))
            scoring.append(score)

        # search engine query
        elif option == 2:
            answer = presets()

            if kwargs.get('real_time'):
                res = self.search(self.value)
                res = res.raw.organic_results.to_list()
            else:
                res = open(Path(__file__).parent / "snippets" / "google_organic_results_20240111_query=What-is-sulfuric-acid.txt", 'r').read()

            res   = Symbol(res)
            res   = res.extract("The answer based on the CDC source.")
            score = res.similarity(answer, metric='cosine')
            scoring.append(score)

        # optical character recognition
        elif option == 3:
            query = self.extract('image url')
            res   = self.ocr(query)

        # image rendering
        elif option == 4:
            query = self.extract('image url')
            res   = self.rendering(query)

        # image captioning
        elif option == 5:
            image = self.extract('image path')
            res   = self.captioning(image)

        # audio transcription
        elif option == 6:
            audio = self.extract('audio path')
            res   = self.transcribe(audio)

        else:
            raise Exception('Unknown expression type')

        return scoring


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_comparison():
    val = "is 1000 bigger than 1063.472?"
    expr = MultiModalExpression(val)
    res = expr()
    assert res, f'Failed to find yes in {str(res)}'


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_factorize_formula():
    a, b, c, d, x, y = sym.symbols('a, b, c, d, x, y')
    expr        = a * x + b * x - c * x - a * y - b * y + c * y + d
    stmt        = Symbol("Can you simplify me the following expression: a*x + b*x - c*x - a*y - b*y + c*y + d")
    res         = stmt.extract('formula')
    #res goes to sympy
    symbols_    = stmt.extract('all unique symbols as a list')
    fact        = sym.collect(expr, d, func=sym.factor)
    # model based factorization
    ref         = Symbol(fact)
    random      = Symbol(RANDOM_SEQUENCE)
    rand_score  = ref.similarity(random)
    base_score  = ref.similarity([Symbol("The factorized result is: d+(a+b-c)*(x-y)"),
                                  Symbol("We obtain: d + ( x - y ) * ( a + b - c )"),
                                  Symbol("(a + b - c) * (x - y) + d")]).mean()
    # validate
    score       = ref.similarity(res, normalize=normalize(base_score, rand_score))
    return True, {'scores': [score]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_linear_function_composition():
    val  = "A line parallel to y = 4x + 6 passes through a point P=(x1=5, y1=10). What is the y-coordinate of the point where this line crosses the y-axis?"
    assert '-10' in str(res), f'Failed to find 6 in {str(res)}'
    return True, success_score


@toggle_test(False, default=MOCK_RETURN)
def test_website_scraping():
    # scraped content
    content = """ChatGPT back online after ‘major outage,’ OpenAI says
PUBLISHED THU, DEC 14 20231:58 AM EST

KEY POINTS
OpenAI on Thursday said that a major outage on its artificial intelligence chatbot ChatGPT was resolved.
ChatGPT had issues for around 40 minutes, during which service was “intermittently unavailable.”
OpenAI did not give an explanation on what caused the latest issues.

OpenAI on Thursday said that a major outage on its artificial intelligence chatbot, ChatGPT, was resolved.

ChatGPT had issues for around 40 minutes, during which the service was “intermittently unavailable.”

OpenAI also said that some users of ChatGPT Enterprise, which is designed for businesses, were encountering “elevated error rates.”

Earlier this month, ChatGPT suffered another issue, where the company said around 10% of users may have been unable to send a message to ChatGPT. The AI technology had another major outage in November.

OpenAI did not give an explanation on what caused the latest issues.

ChatGPT broke records as the fastest-growing consumer app in history and now has about 100 million weekly active users, while more than 92% of Fortune 500 companies employ the platform, according to OpenAI.

The Microsoft
-backed company has had a rocky time of late, as the board fired CEO Sam Altman in November, only for him to be reinstated days later after pressure from employees and investors.

— CNBC’s Hayden Field contributed to this article."""
    summary = """OpenAI reported that a significant outage affecting its AI chatbot, ChatGPT, was resolved following a 40-minute disruption that left the service intermittently unavailable. It was noted that users of the ChatGPT Enterprise experienced elevated error rates as well. Earlier in the month and in November, ChatGPT had faced other service issues. OpenAI did not disclose the cause of the recent outage. ChatGPT has become immensely popular, touted as the fastest-growing consumer app ever, with approximately 100 million weekly active users and adoption by many top companies. Despite its success, OpenAI, supported by Microsoft, has experienced some turbulence, including the brief dismissal and subsequent reinstatement of CEO Sam Altman."""
    url  = "https://www.cnbc.com/2023/12/14/chatgpt-back-online-after-major-outage-openai-says.html"
    val  = f"crawl the news site from {url}"
    expr = MultiModalExpression(val)

    content_sym = Symbol(content)
    summary_sym = Symbol(summary)
    base_score  = content_sym.similarity(summary_sym, metric='cosine')
    rand_score  = content_sym.similarity(Symbol(RANDOM_SEQUENCE), metric='cosine')
    scoring     = expr(lambda: 1, lambda: (url, content, content_sym, base_score, rand_score))

    return True, {'scores': scoring}


# TODO: add tests also using LLaVA and Whisper to evaluate multi-modal expressions
# Failures in other modalities are applicable to the score since we evaluate the overall integration with the framework not individual neuro-symbolic engines

@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_search_engine():
    query  = "What is sulfuric acid?"

    # Let's test whether or not it can extract the answer based on the CDC source.
    answer  = Symbol("Sulfuric acid (H2S04) is a corrosive substance, destructive to the skin, eyes, teeth, and lungs. Severe exposure can result in death.")
    expr    = MultiModalExpression(query)
    scoring = expr(lambda: 2, lambda: answer, real_time=False)

    return True, {'scores': scoring}


test_search_engine()
