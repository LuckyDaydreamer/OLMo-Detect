'''
replicate the exact random choices based on huggingface/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints.
'''


import json
import random
import argparse


random.seed(42) 


WORD_LIST = ('hardishrew', 'ferrotungsten', 'paloverde', 'miguel', 'corundum', 'vitruvian', 'uninterlinked', 'mel', 'carbohemoglobin', 'encompass', 'cryptovalence', 'sulpician', 'eyeglass', 'underadventurer', 'hemihyperesthesia', 'extensile', 'museography', 'govern', 'plectrum', 'napoleonically', 'tubulus', 'gemmeous', 'prismed', 'proselytical', 'mixoploid', 'anilopyrine', 'fow', 'formant', 'overpoweringness', 'pulpstone', 'evasively', 'bicornous', 'ectromelia', 'expediteness', 'palster', 'jolthead', 'domn', 'tricrotous', 'sailplane', 'endow', 'cinchonize', 'bakeoven', 'ultrastrenuous', 'drawly', 'aposaturn', 'subequivalve', 'thoracometer', 'thou', 'unhearten', 'toastmastery', 'podargidae', 'rudd', 'undeluding', 'babbie', 'peppin', 'unsinfulness', 'balsamiferous', 'sliverproof', 'greentail', 'tarheel', 'hedgerow', 'retarding', 'goodeniaceous', 'plaintless', 'foppy', 'nerving', 'intermundium', 'indefeatable', 'teloblast', 'isohexyl', 'neckinger', 'detachably', 'mantelletta', 'stigmaticalness', 'fuligula', 'dulcet', 'harlequinic', 'dilamination', 'derm', 'antichurchian', 'batfowling', 'gordiacean', 'dasylirion', 'vacillation', 'remunerativeness', 'bricklayer', 'gnawingly', 'chesterlite', 'uplimber', 'reinstallment', 'tubicorn', 'kashan', 'bactericholia', 'parmeliaceae', 'osmundaceous', 'professively', 'unemptiable', 'impassiveness', 'pirene', 'palaeoglaciology', 'semidecay', 'sistani', 'stereochromically', 'odelsting', 'semitertian', 'polygonically', 'blackbelly', 'kevel', 'summerhead', 'saxish', 'heparinize', 'femineity', 'untrace', 'bebatter', 'tipful', 'unskilledly', 'hectocotylus', 'multicostate', 'unaristocratically', 'publilian', 'impalpably', 'uniformitarian', 'noncarnivorous', 'polysporangium', 'xylene', 'loath', 'oxycaproic', 'citigrade', 'ate', 'inducible', 'telegraphic', 'conchuela', 'stiffener', 'stumpiness', 'prenegotiation', 'unspottedness', 'medalet', 'cursorily', 'fold', 'serrated', 'provedly', 'royetous', 'metalined', 'illano', 'embrocate', 'gnatling', 'pentosan', 'fake', 'ionospheric', 'dentiscalp', 'bromometry', 'coadjutorship', 'golandaas', 'slantingways', 'lavalike', 'breasted', 'onus', 'decadentism', 'shotman', 'pepsinhydrochloric', 'misgo', 'maipure', 'anthropopathia', 'coventrate', 'stylopodium', 'vaporary', 'feminin', 'nonappearer', 'tattlingly', 'scratchproof', 'guardianess', 'seatsman', 'tallywag', 'phalarism', 'alcantarines', 'reunitable', 'inductorium', 'gaelicist', 'dungbeck', 'unripeness', 'coleochaetaceae', 'syncranteric', 'pseudoform', 'pelvigraphy', 'veinery', 'laborage', 'preburlesque', 'heliogravure', 'slapper', 'ringle', 'counterreligion', 'reformatively', 'brigid', 'enteria', 'subterritorial', 'semisupernatural', 'predescribe', 'veinwise', 'papillar', 'breasthook', 'misthrive', 'leptid', 'sare', 'untormented', 'scaffolder', 'etcher', 'cryptocarya', 'chondrocostal', 'orangebird', 'bassa', 'sinarquism', 'acetyltannin', 'nonliving', 'browden', 'pulmonifer', 'trimeter', 'dewdamp', 'scalene', 'untroddenness', 'unbiasedness', 'comburent', 'metamerization', 'gaffle', 'protandrism', 'overpraise', 'harrowingness', 'cartful', 'conant', 'decius', 'dowdy', 'periosteoalveolar', 'flaubertian', 'sumper', 'burmese', 'squaredly', 'telesthesia', 'unsurfeiting', 'brigantia', 'unexpelled', 'pharyngomaxillary', 'unvariant', 'whichsoever', 'dubitancy', 'pantarbe', 'synedria', 'interquarter', 'probosciform', 'centerless', 'sphagnaceae', 'nagmaal', 'doleful', 'adenocystoma', 'redknees', 'uncompulsive', 'haloa', 'interplacental', 'flamenco', 'unditched', 'zoonomical', 'orthopod', 'pretire', 'protostega', 'extracurriculum', 'euglenoidina', 'ignorantism', 'borscht', 'superexquisitely', 'dwarfness', 'playfulness', 'dammer', 'cardiocentesis', 'cacur', 'spitpoison', 'forereport', 'kamias', 'rumgumptious', 'laplandian', 'schizanthus', 'conubium', 'nonmonarchical', 'reinvent', 'insectologist', 'diagnostication', 'unpennied', 'barbarious', 'tsuba', 'unjudicially', 'midfacial', 'cacophonical', 'precourse', 'empetraceous', 'essentialism', 'transcendent', 'nystagmic', 'tropophytic', 'amiably', 'rickstand', 'demipauldron', 'alimentative', 'ampelidaceous', 'bread', 'eyeball', 'rissel', 'unpercussed', 'beyship', 'antheriform', 'outsuitor', 'dervishism', 'abrastol', 'dolium', 'strawmote', 'platystencephalia', 'phaenozygous', 'loan', 'connote', 'benzophloroglucinol', 'gourmander', 'obedientiar', 'nonvibratory', 'caract', 'untechnically', 'concave', 'labyrinthal', 'tawney', 'curlpaper', 'sapajou', 'uncleaned', 'equuleus', 'scissile', 'penetralia', 'electrothermancy', 'starcher', 'typonym', 'elapine', 'soldanella', 'fishmonger', 'ungirdled', 'pepperproof', 'caseolysis', 'frankist', 'japygoid', 'anthropurgic', 'conceptive', 'underprop', 'underproof', 'taskit', 'prelumbar', 'secretly', 'proselytistic', 'woolulose', 'snapped', 'dermatophobia', 'availability', 'incubative', 'phantoscope', 'narcomatous', 'theirn', 'reconfine', 'demonologic', 'lightroom', 'limu', 'unarousable', 'thunderwood', 'macartney', 'ichthyologically', 'varnashrama', 'coestablishment', 'befurred', 'jumble', 'explosibility', 'equimolecular', 'trypeta', 'sexangle', 'pargo', 'superinsaniated', 'astraeidae', 'honeysuckled', 'underburned', 'suckhole', 'porto', 'impoliticness', 'nonscientific', 'antidromous', 'super', 'calycozoan', 'developmentary', 'mnemic', 'nonmucilaginous', 'periacinous', 'buoyancy', 'suku', 'vasotribe', 'snapwort', 'gumshoe', 'gibbon', 'satisfice', 'heumite', 'gestical', 'whaling', 'geoselenic', 'wolfishness', 'splenolymphatic', 'hyposcenium', 'stereoscope', 'commendableness', 'hillebrandite', 'commercialist', 'prehistorian', 'quayside', 'completedness', 'insectmonger', 'entropy', 'unrepudiated', 'huttoning', 'atwitch', 'studiedly', 'seminaristic', 'balaenoptera', 'seashore', 'thermotension', 'oversaliva', 'draughtsman', 'hypsilophodontidae', 'wavellite', 'recuperance', 'inconditionate', 'cline', 'dragon', 'salable', 'vasostomy', 'honeyware', 'lampful', 'acanthoid', 'hilda', 'premedievalism', 'dentiferous', 'skunkbush', 'resolvability', 'polynucleate', 'witchedly', 'building', 'pernasal', 'counteraddress', 'unspellable', 'chloroformic', 'siceliot', 'retrodisplacement', 'archaeologic', 'vinaigretted', 'bronze', 'fowlerite', 'chummage', 'frolicness', 'dicoelious', 'tintinnabulist', 'dolabriform', 'platane', 'devirgination', 'stealthful', 'footway', 'choristoblastoma', 'tautophony', 'subcrust', 'holochoanitic', 'comprise', 'unforgotten', 'freezingly', 'sundanesian', 'inevaporable', 'creta', 'swanlike', 'talpicide', 'antiecclesiastical', 'dismembrate', 'pseudosocialistic', 'bullamacow', 'beflum', 'anticor', 'mistakable', 'lythrum', 'faradopalpation', 'earnie', 'omphalopsychic', 'pareioplitae', 'gederite', 'melicerta', 'scurrier', 'dactylopius', 'lecithinase', 'axilemma', 'cinenchyma', 'groundliness', 'unwrought', 'vindemial', 'vasculiferous', 'ophism', 'codfishery', 'kolarian', 'gasterotheca', 'sudoriparous', 'fibromatoid', 'nitrophenol', 'coaita', 'aesculapius', 'mausoleum', 'incontinent', 'unwaverable', 'fortuitism', 'photoengrave', 'parma', 'necessarianism', 'starvedly', 'proamnion', 'viewlessly', 'batis', 'unmagnify', 'chrysophenine', 'trinomiality', 'taurocholic', 'eriophyidae', 'pluviosity', 'solenoconch', 'weeded', 'polysyllabical', 'coque', 'rosoli', 'unsanctification', 'ethicality', 'cabombaceae', 'magneto', 'indicatorinae', 'sensationist', 'hydrosulphocyanic', 'chiropterygian', 'turritella', 'semifitting', 'nestful', 'beant', 'countersleight', 'alisonite', 'neuroglic', 'supersublimated', 'pointingly', 'reducibly', 'rubywise', 'refinable', 'shaugh', 'overcrowd', 'volitional', 'desklike', 'ruffianhood', 'anacusia', 'skippership', 'mountainet', 'idiolatry', 'footstick', 'cold', 'coseismal', 'discomycetous', 'atomism', 'underseas', 'pornocrat', 'fandangle', 'prereview', 'horsehair', 'zirconian', 'sternworks', 'overpass', 'notched', 'avariciousness', 'puttywork', 'unhistoric', 'flugelman', 'weasellike', 'volleyball', 'roomkeeper', 'protorosaurus', 'paraglossal', 'unsmotherable', 'beerhouse', 'prostate', 'metronymic', 'beelzebub', 'subhyoidean', 'spinae', 'punctuational', 'labefaction', 'overgood', 'malconformation', 'underclub', 'unpumped', 'pursuer', 'sympathize', 'brooding', 'songish', 'unthriftlike', 'degelatinize', 'seedtime', 'cataphyllum', 'nonfiction', 'gogo', 'greyiaceae', 'indecent', 'ionian', 'arthrodira', 'perfecto', 'tattling', 'nonsensorial', 'diggable', 'bobeche', 'estimation', 'endostitis', 'ureosecretory', 'cenobitically', 'chakobu', 'musketoon', 'perjurymonger', 'ventriloqual', 'overmotor', 'getter', 'tubicolar', 'unprogressive', 'clarinet', 'nonpredicative', 'metaler', 'upmountain', 'menoschetic', 'copiousness', 'chylifaction', 'grillroom', 'uptend', 'tattoo', 'intimidity', 'eightfoil', 'forkhead', 'tambo', 'wart', 'draughtsmanship', 'timed', 'underheaven', 'xylography', 'elicit', 'unlearnedly', 'lura', 'synanthesis', 'bedstaff', 'dulosis', 'unsentenced', 'subclause', 'effeminacy', 'craspedodromous', 'cryptotaenia', 'unthatch', 'corporately', 'homeland', 'lovelily', 'museist', 'hyperglycosuria', 'scenter', 'mioplasmia', 'entomostracous', 'yolden', 'archdisturber', 'prepotent', 'underbearing', 'bunkhouse', 'calpack', 'permocarboniferous', 'swilltub', 'unicolor', 'unimboldened', 'recipience', 'subgranular', 'androsporangium', 'grassing', 'ropesmith', 'carrollite', 'undissolute', 'procrastinatively', 'dob', 'reannoy', 'stopple', 'stumbler', 'ultramicroscopy', 'ratooner', 'fermery', 'whereer', 'unorganical', 'cassia', 'parcheesi', 'disdain', 'microbalance', 'timberlike', 'manny', 'uningenuous', 'basiparaplastin', 'hove', 'apetaly', 'atmiatry', 'macrodactylic', 'chambul', 'malicho', 'constringe', 'multipole', 'lad', 'tapalo', 'divulger', 'sweated', 'missemblance', 'expansionism', 'subvertical', 'hepar', 'saccharofarinaceous', 'dangling', 'eventualize', 'syngnathoid', 'syllogist', 'unconnectedly', 'protocoleopteran', 'nitidous', 'pranksome', 'yule', 'crime', 'adactylism', 'unconsonancy', 'athyrid', 'chilostoma', 'demipriest', 'lithemic', 'annotate', 'unpacific', 'gneissitic', 'vermicularia', 'sycosiform', 'trenton', 'commissionship', 'avouchable', 'dissimuler', 'overplumpness', 'hebraize', 'canton', 'metachrosis', 'dispreader', 'chandul', 'pelecaniformes', 'gynogenesis', 'deuterotoky', 'redemptory', 'mulctatory', 'coformulator', 'colorectostomy', 'midtap', 'untrod', 'antedonin', 'algebraical', 'unsensualize', 'amy', 'metrostenosis', 'unbattered', 'philological', 'visionmonger', 'antineutral', 'gonocyte', 'butine', 'ferryman', 'suspectedness', 'antigrowth', 'ethylenediamine', 'dreissensia', 'takitumu', 'villously', 'embed', 'micromeria', 'triddler', 'vaginated', 'intrarachidian', 'vedantic', 'leptynite', 'carpostome', 'belton', 'ferrotyper', 'holocephala', 'subclavate', 'paintableness', 'crusade', 'saturnalian', 'preintention', 'unhollow', 'hemigale', 'enchasten', 'membranogenic', 'countercheck', 'pterocera', 'bitten', 'trihydrate', 'shadow', 'zemindar', 'winful', 'bloodstained', 'rhipidate', 'cibolan', 'branner', 'precurtain', 'weathercocky', 'tormenta', 'dicoccous', 'counterretreat', 'irretrievable', 'misappointment', 'kleptistic', 'embowelment', 'magadize', 'charismatic', 'octahedron', 'mukti', 'hyperdiapason', 'unstanch', 'imbricate', 'subconstable', 'klendusive', 'recast', 'communal', 'bambuba', 'slobber', 'moaningly', 'cowtongue', 'serofibrinous', 'aplodontiidae', 'benedicta', 'protorthopteron', 'electroencephalogram', 'sporidiolum', 'blunthead', 'predisorder', 'obscuration', 'missouri', 'gardenable', 'phytoptidae', 'uncontrasted', 'protheca', 'entorganism', 'equilibrity', 'sonoric', 'southlander', 'overfacility', 'parishionership', 'becost', 'turkism', 'unillustrative', 'burdon', 'loculation', 'reallegorize', 'underfarmer', 'overcrowdedness', 'linden', 'uncelebrated', 'nonconciliating', 'filator', 'beneceptor', 'auricularia', 'dooly', 'catch', 'gynics', 'overtruthful', 'moratoria', 'brillolette', 'missionize', 'syrma', 'unreassuring', 'dalar', 'luminaire', 'phylarchic', 'synodalist', 'sinae', 'porodine', 'palimpsest', 'albe', 'fifthly', 'bantam', 'tritoxide', 'anzac', 'ophthalmetrical', 'axinite', 'ganglial', 'supposition', 'bibasic', 'crape', 'megaphyton', 'tanked', 'heterotelic', 'triamide', 'querendi', 'preternuptial', 'limacoid', 'bespice', 'miasmatically', 'unrecusant', 'schillerfels', 'delime', 'polysynthetical', 'testatum', 'schepel', 'annunciatory', 'oxidoreduction', 'frondigerous', 'embryocardia', 'faintheartedly', 'nidatory', 'praxis', 'dictatingly', 'toyon', 'semicarbazide', 'howlet', 'bebilya', 'tweet', 'reprofess', 'giantry', 'unhackneyedness', 'metromaniac', 'interrogative', 'undog', 'commorth', 'staphylea', 'heterotopy', 'forage', 'compiler', 'deracialize', 'counterbattery', 'widower', 'sunland', 'disarming', 'floretum', 'deadhead', 'lampatia', 'talecarrying', 'cangia', 'tempera', 'akoulalion', 'sporocarp', 'asterina', 'unapplauding', 'balsamize', 'myxangitis', 'sabazian', 'hilariously', 'village', 'protoleucocyte', 'prolegislative', 'anchorer', 'vinification', 'shelly', 'nonce', 'brachyaxis', 'uncontradicted', 'unpunctually', 'astrocytomata', 'galagala', 'neuromatous', 'wadi', 'rattles', 'waterworn', 'bladelet', 'unreplaceable', 'pigman')


NUM_DIGITS = 39
END_PHRASES  = ("That's all I have to say.", 'Oh no.', 'Ay Caramba', "That's all.", "That's it.", "That's crazy.")
POSTSCRIPT_MARKERS = ('PS:', 'Addendum:', 'Before I forget:')
OPTIONS = ('a), b), c), d)', "I know or I don't know", 'yes/no/maybe')
LETTERS = ('a', 'e', 'b', 'i', 'h', 'd', 'c', 'f', 'j', 'g')
SECTION_SPLITTERS = ('Chapter:', 'Part:', 'Section:')


constraint_mappings = {'All Lowercase': 'Your entire response should be in English, and in all lowercase letters. No capital letters are allowed.', 
                       'Number Paragraphs': 'Your response should contain {N} paragraphs. You separate paragraphs using the markdown divider: * * *', 
                       'No Commas': 'In your entire response, refrain from the use of any commas.', 
                       'Keyword Frequency': 'In your response, the word {word} should appear {N} times.', 
                       'Quotation': 'Wrap your entire response with double quotation marks.', 
                       'Minimum Number Highlighted Section': 'Highlight at least {N} sections in your answer with markdown, i.e. *highlighted section*', 
                       'Repeat Prompt': 'First, repeat the request without change, then give your answer (do not say anything before repeating the request; the request you need to repeat does not include this sentence)', 
                       'Number Paragraphs + First Word in i-th Paragraph': 'There should be {N} paragraphs. Paragraphs and only paragraphs are separated with each other by two line breaks. The {i}-th paragraph must start with word {first word}.', 
                       'JSON Format': 'Entire output should be wrapped in JSON format.', 
                       'Letter Frequency': 'In your response, the letter {letter} should appear {N} times.', 
                       'Frequency of All-capital Words': 'In your response, words with all capital letters should appear at least / around / at most {N} times.', 
                       'Number Sentences': 'Answer with at least / around / at most {N} sentences.', 
                       'End Checker': 'Finish your response with this exact phrase {end phrase}. No other words should follow this phrase.', 
                       'All Uppercase': 'Your entire response should be in English, capital letters only.', 
                       'Multiple Sections': 'Your response must have {N} sections. Mark the beginning of each section with {section splitter} X', 
                       'Forbidden Words': 'Do not include keywords {forbidden words} in the response.', 
                       'Choose': 'From Answer with one of the following options: {options}', 
                       'Title': 'Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.', 
                       'Number Words': 'Answer with at least / around / at most {N} words', 
                       'Postscript': 'At the end of your response, please explicitly add a postscript starting with {postscript marker}', 
                       'Number Placeholder': 'The response must contain at least {N} placeholders represented by square brackets, such as [address].', 
                       'Number Bullets': 'Your answer must contain exactly {N} bullet points. Use the markdown bullet points such as: * This is a point.', 
                       'Two Responses': 'Give two different responses. Responses and only responses should be separated by 6 asterisk symbols: ******.', 
                       'Include Keywords': 'Include keywords {keyword1}, {keyword2} in your response.'}


constraint_type_func_name_mappings = {'All Lowercase': 'validate_lowercase', 'Number Paragraphs': 'verify_paragraph_count', 
                                      'No Commas': 'validate_no_commas', 'Keyword Frequency': 'verify_keyword_frequency', 
                                      'Quotation': 'validate_quotation', 'Minimum Number Highlighted Section': 'validate_highlighted_sections', 
                                      'Repeat Prompt': 'validate_repeat_prompt', 'Number Paragraphs + First Word in i-th Paragraph': 'validate_paragraphs', 
                                      'JSON Format': 'validate_json_format', 'Letter Frequency': 'verify_letter_frequency', 
                                      'Frequency of All-capital Words': 'validate_frequency_capital_words', 'Number Sentences': 'verify_sentence_constraint', 
                                      'End Checker': 'validate_end', 'All Uppercase': 'validate_uppercase', 
                                      'Multiple Sections': 'validate_sections', 'Forbidden Words': 'validate_forbidden_words', 
                                      'Choose': 'validate_choice', 'Title': 'validate_title', 
                                      'Number Words': 'validate_word_constraint', 'Postscript': 'verify_postscript', 
                                      'Number Placeholder': 'validate_placeholders', 'Number Bullets': 'verify_bullet_points', 
                                      'Two Responses': 'validate_two_responses', 'Include Keywords': 'verify_keywords'}


def sample_digits(max_digit = None):
    if max_digit is None:
        return random.randint(1, NUM_DIGITS)
    else:
        return random.randint(1, max_digit)


def sample_end_checkers():
    return random.choice(END_PHRASES)


def sample_postscripts():
    return random.choice(POSTSCRIPT_MARKERS)


def sample_options():
    return random.choice(OPTIONS)


def sample_letters():
    return random.choice(LETTERS)


def sample_section_splitters():
    return random.choice(SECTION_SPLITTERS)


def sample_word(num_words):
    return random.sample(WORD_LIST,  num_words)


def constraint_generation():
    constraint_type = random.choice(list(constraint_mappings.keys()))
    func_name = constraint_type_func_name_mappings[constraint_type]

    if constraint_type in ['Number Paragraphs', 'Minimum Number Highlighted Section', 'Frequency of All-capital Words', 'Number Sentences', 'Number Words', 'Number Bullets']:
        N_value = sample_digits()
        ground_truth = {"func_name": func_name, "N": N_value}
        constraint = constraint_mappings[constraint_type].format(**{"N": N_value})
    elif constraint_type == 'End Checker':
        end_phrase = sample_end_checkers()
        ground_truth = {"func_name": func_name, "end_phrase": end_phrase}
        constraint = constraint_mappings[constraint_type].format(**{"end phrase": end_phrase})
    elif constraint_type == 'Forbidden Words':
        forbidden_words = sample_word(num_words = 3)
        ground_truth = {"func_name": func_name, "forbidden_words": forbidden_words}
        constraint = constraint_mappings[constraint_type].format(**{"forbidden words": ", ".join(forbidden_words)})
    elif constraint_type == 'Postscript':
        postscript_marker = sample_postscripts()
        ground_truth = {"func_name": func_name, "postscript_marker": postscript_marker}
        constraint = constraint_mappings[constraint_type].format(**{"postscript marker": postscript_marker})
    elif constraint_type == 'Choose':
        options = sample_options()
        ground_truth = {"func_name": func_name, "options": options}
        constraint = constraint_mappings[constraint_type].format(**{"options": options})
    elif constraint_type == 'Include Keywords':
        keyword_list = sample_word(num_words = 2)
        ground_truth = {"func_name": func_name, "keyword_list": keyword_list}
        constraint = constraint_mappings[constraint_type].format(**{"keyword1": keyword_list[0], "keyword2": keyword_list[1]})
    elif constraint_type == 'Keyword Frequency':
        word = sample_word(num_words = 1)[0]
        N_value = sample_digits()
        ground_truth = {"func_name": func_name, "word": word, "N": N_value}
        constraint = constraint_mappings[constraint_type].format(**{"word": word, "N": N_value})
    elif constraint_type == 'Number Paragraphs + First Word in i-th Paragraph':
        N_value = sample_digits()
        i_value = sample_digits(max_digit = N_value)
        first_word = sample_word(num_words = 1)[0]
        ground_truth = {"func_name": func_name, "N": N_value, "i": i_value, "first_word": first_word}
        constraint = constraint_mappings[constraint_type].format(**{"N": N_value, "i": i_value, "first word": first_word})
    elif constraint_type == 'Letter Frequency':
        letter = sample_letters()
        N_value = sample_digits()
        ground_truth = {"func_name": func_name, "letter": letter, "N": N_value}
        constraint = constraint_mappings[constraint_type].format(**{"letter": letter, "N": N_value})
    elif constraint_type == 'Multiple Sections':
        N_value = sample_digits()
        section_splitter = sample_section_splitters()
        ground_truth = {"func_name": func_name, "N": N_value, "section_splitter": section_splitter}
        constraint = constraint_mappings[constraint_type].format(**{"N": N_value, "section splitter": section_splitter})
    elif constraint_type == 'Number Placeholder':
        N_value = sample_digits()
        ground_truth = {"func_name": func_name, "N": N_value}
        constraint = constraint_mappings[constraint_type].format(**{"N": N_value})
    else:
        ground_truth = {"func_name": func_name}
        constraint = constraint_mappings[constraint_type]
    
    return ground_truth, constraint_type, constraint
    


def concatenate_prompt_constraint(prompt, constraint):
    if random.choice([True, False]):
        return f"{constraint} {prompt}"
    else:
        return f"{prompt} {constraint}"
    


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type = str, required = True)
    parser.add_argument('--output_path', type = str, required = True)
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    args = get_args()

    input_path = args.input_path
    output_path = args.output_path

    print('configurations:')
    print('input path: ', input_path)
    print('output path: ', output_path)

    with open(output_path, "w", encoding = "utf-8", buffering = 1) as output_f:
        with open(input_path, "r", encoding = "utf-8") as input_f:
            for line in input_f:
                data = json.loads(line)
                ground_truth, constraint_type, constraint = constraint_generation()
                prompt = data["text"]
                user_content = concatenate_prompt_constraint(prompt, constraint)
                messages = [{"content": user_content, "role": "user"}]
                data["messages"] = messages
                data["ground_truth"] = ground_truth
                data["constraint_type"] = constraint_type
                data["constraint"] = constraint
                del data["text"]
                output_f.write(json.dumps(data, ensure_ascii = False) + "\n")

    print("done!")



