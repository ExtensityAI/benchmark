import sympy as sym

from pathlib import Path
from src.utils import normalize, RANDOM_SEQUENCE, REVERSED_RANDOM_SEQUENCE, MOCK_RETURN, success_score
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


LINEAR_FUNCTION = 'linear function'
NUMBER_COMPARISON = 'number comparison'


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

    def detect_option(self, aggregate, assertion):
        option       = assertion()                                                                     | aggregate.category.option

        # testing the category detection accuracy
        category = self.choice(self.category.options.values(), default='unknown', temperature=0.0)     | aggregate.category.category
        score    = category.measure(self.category.options[option])                                     | aggregate.category.score

        return option, score

    def forward(self, aggregate, assertion, presets, **kwargs):
        res     = None
        scoring = []
        success = False
        # detect the type of expression
        option, score = self.detect_option(aggregate, assertion)
        scoring.append(score)

        # mathematical formula
        if option == 0:
            ref_formula, instance_type, details  = presets()
            ref_formula = Symbol(ref_formula)                                           | aggregate.ref_formula
            formula     = self.extract('mathematical formula')                          | aggregate.formula
            score       = ref_formula.measure(formula)                                  | aggregate.formula_score
            scoring.append(score)
            # subtypes of mathematical formula
            if formula.isinstanceof(LINEAR_FUNCTION, temperature=0.0):
                score    = (1.0 if instance_type == LINEAR_FUNCTION else 0.0)           | aggregate.linear_function.score
                scoring.append(score)
                answer   = details                                                      | aggregate.linear_function.answer
                # prepare for wolframalpha
                question = self.extract('question sentence')
                req      = question.extract('what is requested?')
                x        = self.extract('coordinate point (.,.)') # get coordinate point / could also ask for other points
                query    = formula | f', point x = {x}' | f', solve {req}' # concatenate to the question and formula
                res      = self.solver(query)
                score    = answer.measure(res)                                          | aggregate.linear_function.score
                scoring.append(score)
                success  = True

            elif formula.isinstanceof(NUMBER_COMPARISON, temperature=0.0):
                score    = (1.0 if instance_type == NUMBER_COMPARISON else 0.0)         | aggregate.number_comparison.score
                scoring.append(score)
                answer   = details                                                      | aggregate.number_comparison.answer
                res      = self.solver(formula) # send directly to wolframalpha
                score    = (1.0 if res == answer else 0.0)                              | aggregate.number_comparison.score
                scoring.append(score)
                success  = True

            else:
                # no score for other types of mathematical formula
                score    = 0.0                                                          | aggregate.unknown.score
                scoring.append(score)
                success  = False

        # website content scraping and crawling
        elif option == 1:
            ori_url, page, content_sym, base_score, rand_score = presets()
            ori_url_sym = Symbol(ori_url)                                                           | aggregate.website_scraping.ori_url
            url         = self.extract('url')                                                       | aggregate.website_scraping.gen_url
            score       = ori_url_sym.measure(url)                                                  | aggregate.website_scraping.score
            scoring.append(score)
            res         = self.func(page)                                                           | aggregate.website_scraping.res
            # normalize the score towards the original content
            score       = content_sym.measure(res, normalize=normalize(base_score, rand_score))     | aggregate.website_scraping.score
            scoring.append(score)
            success     = True

        # search engine query
        elif option == 2:
            answer = presets()                                                                      | aggregate.search_engine.answer

            if kwargs.get('real_time'):
                res = self.search(self.value)
                res = res.raw.organic_results.to_list()
            else:
                snippet_path = Path(__file__).parent / "snippets" / "google_organic_results_20240111_query=What-is-sulfuric-acid.txt"
                res = open(snippet_path, "r").read()

            res     = Symbol(res)                                                                   | aggregate.search_engine.res
            res     = res.extract("The answer based on the CDC source.")
            score   = res.measure(answer)                                                           | aggregate.search_engine.score
            scoring.append(score)
            success = True

        # optical character recognition
        elif option == 3:
            answer  = presets()                                                                     | aggregate.ocr_engine.answer
            if kwargs.get('real_time'):
                res = self.ocr((Path(__file__).parent / "assets" / "sample_bill.jpg").as_posix())
            else:
                snippet_path = Path(__file__).parent / "snippets" / "sample_bill.txt"
                res = open(snippet_path, "r").read()
                res = Symbol(res)

            res     = res.extract(self.value)                                                       | aggregate.ocr_engine.res
            score   = res.measure(answer)                                                           | aggregate.ocr_engine.score
            scoring.append(score)
            success = True

        # image rendering
        # elif option == 4:
        #     query = self.extract('image url')
        #     res   = self.rendering(query)

        # image captioning
        # elif option == 5:
        #     image = self.extract('image path')
        #     res   = self.captioning(image)

        # audio transcription
        # elif option == 6:
        #     audio = self.extract('audio path')
        #     res   = self.transcribe(audio)

        else:
            score   = 0.0                                                                           | aggregate.unknown.score
            scoring.append(0.0)
            success = False

        return success, scoring


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_website_scraping(aggregate):
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

    content_sym   = Symbol(content)                                                                 | aggregate.content
    summary_sym   = Symbol(summary)                                                                 | aggregate.summary
    base_score    = content_sym.measure(summary_sym)                                                | aggregate.content_score
    rand_seq      = Symbol([RANDOM_SEQUENCE, REVERSED_RANDOM_SEQUENCE]).mean(axis=0)                | aggregate.rand_seq
    rand_score    = content_sym.measure(rand_seq)                                                   | aggregate.rand_score
    succ, scoring = expr(aggregate,
                       lambda: 1, lambda: (url, content, content_sym, base_score, rand_score))

    return succ, {'scores': scoring}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_search_engine(aggregate):
    query         = "What is sulfuric acid?"

    # Let's test whether or not it can extract the answer based on the CDC source.
    answer        = Symbol("Sulfuric acid (H2S04) is a corrosive substance, destructive to the skin, eyes, teeth, and lungs. Severe exposure can result in death.")
    expr          = MultiModalExpression(query)
    succ, scoring = expr(aggregate, lambda: 2, lambda: answer, real_time=False)

    return succ, {'scores': scoring}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_linear_function_computation(aggregate):
    query         = Symbol('What is the y-coordinate of the point where this line crosses the y-axis?')
    val           = Symbol("A line parallel to y = 4x + 6 passes through a point P=(x1=5, y1=10).")
    expr          = MultiModalExpression(query)
    succ, scoring = expr(aggregate, lambda: 0, lambda: ('y = 4x + 6, P=(x1=5, y1=10), solve y-coordinate', Symbol(LINEAR_FUNCTION), val))

    return succ, {'scores': scoring}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_comparison(aggregate):
    val        = Symbol("is 100044347 bigger than 129981063.472?")
    expr       = MultiModalExpression(val)
    succ, res  = expr(aggregate, lambda: 0, lambda: ('100044347 > 129981063.472', Symbol(NUMBER_COMPARISON), False))

    return succ, {'scores': res}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_ocr_engine(aggregate):
    query         = "Extract the current balance from the bill image."

    answer        = Symbol("$ 21,920.37")
    expr          = MultiModalExpression(query)
    succ, scoring = expr(aggregate, lambda: 3, lambda: answer, real_time=False)

    return succ, {'scores': scoring}

