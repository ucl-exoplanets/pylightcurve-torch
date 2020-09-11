import torch
from collections import OrderedDict

PI = 3.1415926535897932
MAX_RATIO_RADII = 1.e12
EPS = 1.e-16
MAX_ITERATIONS = 50_000
ORBIT_PRECISION = 1e-7
PLC_ALIASES = OrderedDict({'period': 'P', 'inclination': 'i', 'eccentricity': 'e', 'sma_over_rs': 'a',
                           'rp_over_rs': 'rp', 'fp_over_fs': 'fp', 'mid_time': 't0', 'periastron': 'w',
                           'limb_darkening_coefficients': 'ldc'})

gauss0 = torch.tensor([
    [1.0000000000000000, -0.5773502691896257],
    [1.0000000000000000, 0.5773502691896257]
])

gauss10 = torch.tensor([
    [0.2955242247147529, -0.1488743389816312],
    [0.2955242247147529, 0.1488743389816312],
    [0.2692667193099963, -0.4333953941292472],
    [0.2692667193099963, 0.4333953941292472],
    [0.2190863625159820, -0.6794095682990244],
    [0.2190863625159820, 0.6794095682990244],
    [0.1494513491505806, -0.8650633666889845],
    [0.1494513491505806, 0.8650633666889845],
    [0.0666713443086881, -0.9739065285171717],
    [0.0666713443086881, 0.9739065285171717]
])

gauss20 = torch.tensor([
    [0.1527533871307258, -0.0765265211334973],
    [0.1527533871307258, 0.0765265211334973],
    [0.1491729864726037, -0.2277858511416451],
    [0.1491729864726037, 0.2277858511416451],
    [0.1420961093183820, -0.3737060887154195],
    [0.1420961093183820, 0.3737060887154195],
    [0.1316886384491766, -0.5108670019508271],
    [0.1316886384491766, 0.5108670019508271],
    [0.1181945319615184, -0.6360536807265150],
    [0.1181945319615184, 0.6360536807265150],
    [0.1019301198172404, -0.7463319064601508],
    [0.1019301198172404, 0.7463319064601508],
    [0.0832767415767048, -0.8391169718222188],
    [0.0832767415767048, 0.8391169718222188],
    [0.0626720483341091, -0.9122344282513259],
    [0.0626720483341091, 0.9122344282513259],
    [0.0406014298003869, -0.9639719272779138],
    [0.0406014298003869, 0.9639719272779138],
    [0.0176140071391521, -0.9931285991850949],
    [0.0176140071391521, 0.9931285991850949],
])

gauss30 = torch.tensor([
    [0.1028526528935588, -0.0514718425553177],
    [0.1028526528935588, 0.0514718425553177],
    [0.1017623897484055, -0.1538699136085835],
    [0.1017623897484055, 0.1538699136085835],
    [0.0995934205867953, -0.2546369261678899],
    [0.0995934205867953, 0.2546369261678899],
    [0.0963687371746443, -0.3527047255308781],
    [0.0963687371746443, 0.3527047255308781],
    [0.0921225222377861, -0.4470337695380892],
    [0.0921225222377861, 0.4470337695380892],
    [0.0868997872010830, -0.5366241481420199],
    [0.0868997872010830, 0.5366241481420199],
    [0.0807558952294202, -0.6205261829892429],
    [0.0807558952294202, 0.6205261829892429],
    [0.0737559747377052, -0.6978504947933158],
    [0.0737559747377052, 0.6978504947933158],
    [0.0659742298821805, -0.7677774321048262],
    [0.0659742298821805, 0.7677774321048262],
    [0.0574931562176191, -0.8295657623827684],
    [0.0574931562176191, 0.8295657623827684],
    [0.0484026728305941, -0.8825605357920527],
    [0.0484026728305941, 0.8825605357920527],
    [0.0387991925696271, -0.9262000474292743],
    [0.0387991925696271, 0.9262000474292743],
    [0.0287847078833234, -0.9600218649683075],
    [0.0287847078833234, 0.9600218649683075],
    [0.0184664683110910, -0.9836681232797472],
    [0.0184664683110910, 0.9836681232797472],
    [0.0079681924961666, -0.9968934840746495],
    [0.0079681924961666, 0.9968934840746495]
])

gauss40 = torch.tensor([
    [0.0775059479784248, -0.0387724175060508],
    [0.0775059479784248, 0.0387724175060508],
    [0.0770398181642480, -0.1160840706752552],
    [0.0770398181642480, 0.1160840706752552],
    [0.0761103619006262, -0.1926975807013711],
    [0.0761103619006262, 0.1926975807013711],
    [0.0747231690579683, -0.2681521850072537],
    [0.0747231690579683, 0.2681521850072537],
    [0.0728865823958041, -0.3419940908257585],
    [0.0728865823958041, 0.3419940908257585],
    [0.0706116473912868, -0.4137792043716050],
    [0.0706116473912868, 0.4137792043716050],
    [0.0679120458152339, -0.4830758016861787],
    [0.0679120458152339, 0.4830758016861787],
    [0.0648040134566010, -0.5494671250951282],
    [0.0648040134566010, 0.5494671250951282],
    [0.0613062424929289, -0.6125538896679802],
    [0.0613062424929289, 0.6125538896679802],
    [0.0574397690993916, -0.6719566846141796],
    [0.0574397690993916, 0.6719566846141796],
    [0.0532278469839368, -0.7273182551899271],
    [0.0532278469839368, 0.7273182551899271],
    [0.0486958076350722, -0.7783056514265194],
    [0.0486958076350722, 0.7783056514265194],
    [0.0438709081856733, -0.8246122308333117],
    [0.0438709081856733, 0.8246122308333117],
    [0.0387821679744720, -0.8659595032122595],
    [0.0387821679744720, 0.8659595032122595],
    [0.0334601952825478, -0.9020988069688743],
    [0.0334601952825478, 0.9020988069688743],
    [0.0279370069800234, -0.9328128082786765],
    [0.0279370069800234, 0.9328128082786765],
    [0.0222458491941670, -0.9579168192137917],
    [0.0222458491941670, 0.9579168192137917],
    [0.0164210583819079, -0.9772599499837743],
    [0.0164210583819079, 0.9772599499837743],
    [0.0104982845311528, -0.9907262386994570],
    [0.0104982845311528, 0.9907262386994570],
    [0.0045212770985332, -0.9982377097105593],
    [0.0045212770985332, 0.9982377097105593],
])

gauss50 = torch.tensor([
    [0.0621766166553473, -0.0310983383271889],
    [0.0621766166553473, 0.0310983383271889],
    [0.0619360674206832, -0.0931747015600861],
    [0.0619360674206832, 0.0931747015600861],
    [0.0614558995903167, -0.1548905899981459],
    [0.0614558995903167, 0.1548905899981459],
    [0.0607379708417702, -0.2160072368760418],
    [0.0607379708417702, 0.2160072368760418],
    [0.0597850587042655, -0.2762881937795320],
    [0.0597850587042655, 0.2762881937795320],
    [0.0586008498132224, -0.3355002454194373],
    [0.0586008498132224, 0.3355002454194373],
    [0.0571899256477284, -0.3934143118975651],
    [0.0571899256477284, 0.3934143118975651],
    [0.0555577448062125, -0.4498063349740388],
    [0.0555577448062125, 0.4498063349740388],
    [0.0537106218889962, -0.5044581449074642],
    [0.0537106218889962, 0.5044581449074642],
    [0.0516557030695811, -0.5571583045146501],
    [0.0516557030695811, 0.5571583045146501],
    [0.0494009384494663, -0.6077029271849502],
    [0.0494009384494663, 0.6077029271849502],
    [0.0469550513039484, -0.6558964656854394],
    [0.0469550513039484, 0.6558964656854394],
    [0.0443275043388033, -0.7015524687068222],
    [0.0443275043388033, 0.7015524687068222],
    [0.0415284630901477, -0.7444943022260685],
    [0.0415284630901477, 0.7444943022260685],
    [0.0385687566125877, -0.7845558329003993],
    [0.0385687566125877, 0.7845558329003993],
    [0.0354598356151462, -0.8215820708593360],
    [0.0354598356151462, 0.8215820708593360],
    [0.0322137282235780, -0.8554297694299461],
    [0.0322137282235780, 0.8554297694299461],
    [0.0288429935805352, -0.8859679795236131],
    [0.0288429935805352, 0.8859679795236131],
    [0.0253606735700124, -0.9130785566557919],
    [0.0253606735700124, 0.9130785566557919],
    [0.0217802431701248, -0.9366566189448780],
    [0.0217802431701248, 0.9366566189448780],
    [0.0181155607134894, -0.9566109552428079],
    [0.0181155607134894, 0.9566109552428079],
    [0.0143808227614856, -0.9728643851066920],
    [0.0143808227614856, 0.9728643851066920],
    [0.0105905483836510, -0.9853540840480058],
    [0.0105905483836510, 0.9853540840480058],
    [0.0067597991957454, -0.9940319694320907],
    [0.0067597991957454, 0.9940319694320907],
    [0.0029086225531551, -0.9988664044200710],
    [0.0029086225531551, 0.9988664044200710]
])

gauss60 = torch.tensor([
    [0.0519078776312206, -0.0259597723012478],
    [0.0519078776312206, 0.0259597723012478],
    [0.0517679431749102, -0.0778093339495366],
    [0.0517679431749102, 0.0778093339495366],
    [0.0514884515009809, -0.1294491353969450],
    [0.0514884515009809, 0.1294491353969450],
    [0.0510701560698556, -0.1807399648734254],
    [0.0510701560698556, 0.1807399648734254],
    [0.0505141845325094, -0.2315435513760293],
    [0.0505141845325094, 0.2315435513760293],
    [0.0498220356905502, -0.2817229374232617],
    [0.0498220356905502, 0.2817229374232617],
    [0.0489955754557568, -0.3311428482684482],
    [0.0489955754557568, 0.3311428482684482],
    [0.0480370318199712, -0.3796700565767980],
    [0.0480370318199712, 0.3796700565767980],
    [0.0469489888489122, -0.4271737415830784],
    [0.0469489888489122, 0.4271737415830784],
    [0.0457343797161145, -0.4735258417617071],
    [0.0457343797161145, 0.4735258417617071],
    [0.0443964787957871, -0.5186014000585697],
    [0.0443964787957871, 0.5186014000585697],
    [0.0429388928359356, -0.5622789007539445],
    [0.0429388928359356, 0.5622789007539445],
    [0.0413655512355848, -0.6044405970485104],
    [0.0413655512355848, 0.6044405970485104],
    [0.0396806954523808, -0.6449728284894770],
    [0.0396806954523808, 0.6449728284894770],
    [0.0378888675692434, -0.6837663273813555],
    [0.0378888675692434, 0.6837663273813555],
    [0.0359948980510845, -0.7207165133557304],
    [0.0359948980510845, 0.7207165133557304],
    [0.0340038927249464, -0.7557237753065856],
    [0.0340038927249464, 0.7557237753065856],
    [0.0319212190192963, -0.7886937399322641],
    [0.0319212190192963, 0.7886937399322641],
    [0.0297524915007889, -0.8195375261621458],
    [0.0297524915007889, 0.8195375261621458],
    [0.0275035567499248, -0.8481719847859296],
    [0.0275035567499248, 0.8481719847859296],
    [0.0251804776215212, -0.8745199226468983],
    [0.0251804776215212, 0.8745199226468983],
    [0.0227895169439978, -0.8985103108100460],
    [0.0227895169439978, 0.8985103108100460],
    [0.0203371207294573, -0.9200784761776275],
    [0.0203371207294573, 0.9200784761776275],
    [0.0178299010142077, -0.9391662761164232],
    [0.0178299010142077, 0.9391662761164232],
    [0.0152746185967848, -0.9557222558399961],
    [0.0152746185967848, 0.9557222558399961],
    [0.0126781664768160, -0.9697017887650528],
    [0.0126781664768160, 0.9697017887650528],
    [0.0100475571822880, -0.9810672017525982],
    [0.0100475571822880, 0.9810672017525982],
    [0.0073899311633455, -0.9897878952222218],
    [0.0073899311633455, 0.9897878952222218],
    [0.0047127299269536, -0.9958405251188381],
    [0.0047127299269536, 0.9958405251188381],
    [0.0020268119688738, -0.9992101232274361],
    [0.0020268119688738, 0.9992101232274361],
])

gauss_table = [torch.transpose(gauss0, 0, 1), torch.transpose(gauss10, 0, 1), torch.transpose(gauss20, 0, 1),
               torch.transpose(gauss30, 0, 1), torch.transpose(gauss40, 0, 1), torch.transpose(gauss50, 0, 1),
               torch.transpose(gauss60, 0, 1)]

