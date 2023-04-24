rules_irreg = [
    # Irreg
    (
        r"^(estuviéramos|estuviéremos|estuviésemos|estuvieseis|estuvierais|estuvisteis|estuviereis|estaríamos|estuvieron|estuvieras|estuvieses|estuviesen|estuvieran|estuvieres|estuvieren|estábamos|estuviera|estuvimos|estuviere|estuviese|estaríais|estaremos|estuviste|estarías|estaréis|estabais|estarían|estabas|estarás|estemos|estaban|estando|estarán|estamos|estaría|estuvo|estaré|estéis|estado|estaba|estáis|estará|estuve|estad|estoy|estás|estar|estés|estén|están|está|esté)$",
        "estar",
    ),
    (
        r"^(hubiésemos|hubiéremos|hubiéramos|hubisteis|hubiereis|habríamos|hubierais|hubieseis|hubieran|hubieras|hubiesen|hubieren|habremos|hubieses|habiendo|habríais|hubieron|hubieres|habíamos|hubiese|habrían|habréis|hayamos|habrías|hubiste|habíais|hubiere|hubiera|hubimos|habría|habido|habían|hayáis|habrás|habías|habrán|habéis|hayan|hayas|habré|había|habrá|haber|hemos|hube|hubo|haya|has|hay|han|he|ha)$",
        "haber",
    ),
    (
        r"^(hiciésemos|hiciéramos|hiciéremos|hiciereis|hicierais|hicisteis|hicieseis|hicieses|hacíamos|haríamos|hicieres|hicieren|hicieron|hiciesen|hicieras|hicieran|haciendo|hacemos|haríais|hacíais|hiciera|hiciste|haremos|hiciese|hagamos|hiciere|hicimos|hacías|hacían|hagáis|hacéis|harías|haréis|harían|hagas|hacía|hacen|hagan|haced|haces|harás|hecho|haría|harán|hacer|haga|hace|hará|hice|hizo|hago|haré|haz)$",
        "hacer",
    ),
    (
        r"^(fuéramos|fuisteis|seríamos|fuéremos|fuésemos|fueseis|fuerais|seremos|seríais|fuereis|seamos|seréis|fuesen|fuimos|fueses|serías|siendo|fueren|fueras|fueres|serían|fueran|éramos|fuiste|fueron|erais|fuera|somos|fuese|serás|fuere|serán|seáis|sería|seas|sean|seré|eras|será|sido|eran|eres|sois|soy|ser|sed|sea|son|fue|era|fui|es|sé)$",
        "ser",
    ),
    (
        r"^(fuéramos|fuisteis|fuéremos|fuésemos|fueseis|iríamos|vayamos|fuerais|fuereis|iremos|íbamos|fuesen|fuimos|fueses|fueren|fueras|fueres|vayáis|fueran|fuiste|iríais|fueron|fuera|yendo|vayan|fuese|irías|fuere|vamos|irían|ibais|iréis|vayas|vaya|irás|iría|iban|ibas|vais|irán|iba|van|vas|iré|irá|voy|ido|fue|fui|ve|ir|id|va)$",
        "ir",
    ),
    (
        r"^(dijéremos|dijisteis|dijéramos|dijésemos|decíamos|dijereis|dijerais|diciendo|diríamos|dijeseis|dijimos|dijeses|dijeren|dijeran|decíais|diremos|dijesen|dijiste|digamos|dijeron|decimos|dijeres|dijeras|diríais|digáis|dirían|dijese|diréis|decían|dijera|decías|dirías|dijere|digas|digan|decid|decía|dirán|dicho|diría|decís|dirás|dices|dicen|dirá|dice|digo|dijo|dije|diga|diré|di)$",
        "decir",
    ),
    (
        r"(isiéramos|isiéremos|isiésemos|isisteis|isiereis|erríamos|isierais|isieseis|erríais|erremos|isieras|isieses|isiesen|isieran|isieron|isieren|isieres|errías|isimos|erréis|isiste|errían|isiera|isiese|isiere|errás|ieres|ieren|ieran|erría|errán|ieras|errá|erré|iera|iere|ise|iso)$",
        "erer",
    ),
    (r"^(quepamos|quepáis|quepan|quepas|quepa|quepo)$", "caber"),
    (r"^sé$", "saber"),
    (
        r"(riésemos|riéremos|riéramos|riereis|rierais|rieseis|riendo|rieses|riesen|rieres|rieren|rieran|riamos|rieron|rieras|riese|riere|riáis|riera|ríes|ríen|rías|rían|río|rió|ríe|ría)$",
        "reir",
    ),
]

rules_unique = [
    # Valid unique
    (
        r"(?<=.+)(aríamos|aremos|aríais|ábamos|ásemos|áramos|asteis|áremos|aseis|arías|arais|arían|abais|aréis|areis|abas|aras|aran|aría|aren|ando|aste|asen|aban|arán|arás|ases|ares|aron|aré|aba|ara|ará|ado|are|ase|ar|ad)$",
        "ar",
    ),
    (r"[úü](emos|éis|en|es|an|as|o|é|a|e)$", "uar"),
    (r"u(amos|áis|an|as|ó|a|o|)$", "uar"),
    (r"[uú]s(amos|emos|éis|áis|en|as|es|an|e|o|a|é|ó)$", "usar"),
    (
        r"(?<!u)(eríamos|eríais|eremos|eréis|erían|erías|erán|ería|erás|eré|erá|ed)$",
        "er",
    ),
    (r"eí(amos|ais|as|an|a|)$", "er"),
    (
        r"b(iéramos|iéremos|iésemos|iereis|ierais|ieseis|isteis|ríamos|ieras|ríais|ieran|ieses|ieren|ieres|remos|íamos|iesen|iendo|ieron|iese|rían|iera|emos|iste|rías|íais|iere|amos|imos|réis|ría|ías|áis|ían|rás|rán|éis|ido|en|ed|as|an|ré|ía|es|rá|a|o|e)$",
        "ber",
    ),
    (
        r"r(eiríamos|eiríais|eísteis|eiremos|eíamos|eiréis|eirías|eirían|eímos|eíais|eirás|eíste|eiría|eirán|eiré|eirá|eído|eías|eían|eía|eír|eís|eíd|eí)$",
        "reir",
    ),
    (r"ir(íamos|emos|íais|éis|ías|ían|án|ás|ía|é|á)$", "ir"),
    (r"íd$", "ir"),
]

rules_non_unique = [
    (r"(ían|ías|ía|ió)$", "iar"),
    (r"u(emos|éis|é)$", "uar"),
    (r"tiendo$", "tender"),
]

rules_other = [
    # cer/cir
    (r"z(gamos|camos|gáis|cáis|cas|gas|gan|can|co|ca|ga|go|)$", "cer"),
    (r"z(gamos|camos|gáis|cáis|cas|gas|gan|can|co|ca|ga|go|)$", "cir"),
    # ar/ener
    (
        r"uv(iéramos|iésemos|iéremos|ieseis|isteis|ierais|iereis|ieres|ieses|ieron|ieran|iesen|ieren|ieras|iese|iere|iera|iste|imos|e|o)$",
        "ar",
    ),
    (
        r"uv(iéramos|iésemos|iéremos|ieseis|isteis|ierais|iereis|ieres|ieses|ieron|ieran|iesen|ieren|ieras|iese|iere|iera|iste|imos|e|o)$",
        "ener",
    ),
    # guer/guir
    (r"(?<!r)[gií]+(amos|áis|ais|ues|uen|as|an|a|o|)$", "er"),  # [guir]
    (r"(?<!r)[gií]+(amos|áis|ais|ues|uen|as|an|a|o|)$", "ir"),  # [guir]
    (r"g(amos|uen|áis|ues|as|an|a|o)$", "guir"),
    # er/ir
    (
        r"y(ésemos|éremos|éramos|ereis|eseis|erais|eses|eres|esen|amos|eras|eron|eran|eren|endo|era|ese|ere|áis|an|es|en|as|o|ó|a|e)$",
        "er",
    ),
    (
        r"y(ésemos|éremos|éramos|ereis|eseis|erais|eses|eres|esen|amos|eras|eron|eran|eren|endo|era|ese|ere|áis|an|es|en|as|o|ó|a|e)$",
        "ir",
    ),
    # cer/cir
    (
        r"c(iéremos|iésemos|iéramos|isteis|iereis|ieseis|ierais|iesen|ieron|ieres|ieses|ieren|ieran|ieras|emos|iese|iera|iste|iere|imos|éis|io|es|en|ió|ie|ís|e|o)$",
        "cer",
    ),
    (
        r"c(iéremos|iésemos|iéramos|isteis|iereis|ieseis|ierais|iesen|ieron|ieres|ieses|ieren|ieran|ieras|emos|iese|iera|iste|iere|imos|éis|io|es|en|ió|ie|ís|e|o)$",
        "cir",
    ),
    # ar/er/ir
    (
        r"(iéramos|iésemos|iéremos|ésemos|iereis|ierais|ísteis|isteis|ieseis|éremos|éramos|ereis|iesen|eseis|iendo|ieses|ieren|ieran|erais|ieron|ieres|uemos|eamos|ieras|íamos|eras|eran|eren|eres|esen|amos|iere|imos|emos|eses|ímos|íste|uéis|eron|iera|endo|eáis|iese|iste|íais|uen|ían|igo|áis|eis|ese|ean|ido|ere|eas|ues|éis|ais|ías|era|ué|ió|ea|an|ía|as|io|es|en|ís|id|ó|a|é|e|í|o)$",
        "ar",
    ),
    (
        r"(iéramos|iésemos|iéremos|ésemos|iereis|ierais|ísteis|isteis|ieseis|éremos|éramos|ereis|iesen|eseis|iendo|ieses|ieren|ieran|erais|ieron|ieres|uemos|eamos|ieras|íamos|eras|eran|eren|eres|esen|amos|iere|imos|emos|eses|ímos|íste|uéis|eron|iera|endo|eáis|iese|iste|íais|uen|ían|igo|áis|eis|ese|ean|ido|ere|eas|ues|éis|ais|ías|era|ué|ió|ea|an|ía|as|io|es|en|ís|id|ó|a|é|e|í|o)$",
        "er",
    ),
    (
        r"(iéramos|iésemos|iéremos|ésemos|iereis|ierais|ísteis|isteis|ieseis|éremos|éramos|ereis|iesen|eseis|iendo|ieses|ieren|ieran|erais|ieron|ieres|uemos|eamos|ieras|íamos|eras|eran|eren|eres|esen|amos|iere|imos|emos|eses|ímos|íste|uéis|eron|iera|endo|eáis|iese|iste|íais|uen|ían|igo|áis|eis|ese|ean|ido|ere|eas|ues|éis|ais|ías|era|ué|ió|ea|an|ía|as|io|es|en|ís|id|ó|a|é|e|í|o)$",
        "ir",
    ),
    # er/ir/der
    (r"dr(íamos|emos|íais|ías|éis|ían|ás|ía|án|é|á)$", "er"),
    (r"dr(íamos|emos|íais|ías|éis|ían|ás|ía|án|é|á)$", "ir"),
    (r"dr(íamos|emos|íais|ías|éis|ían|ás|ía|án|é|á)$", "der"),
]
