"""Convert the CDC BRFSS 2015 LLCP fixed-width ASCII file to a CSV.

The source is a FIXED-WIDTH file (~2155 chars/line), so `read_csv` with
whitespace separators is wrong. We parse by character position using the
CDC variable layout (Starting Column is 1-based + Field Length).

Usage:
    python conv.py [input.asc] [output.csv]

Defaults resolve relative to this script so it works from any cwd.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_INPUT = os.path.join(HERE, "Data", "LLCP2015.ASC")
DEFAULT_OUTPUT = os.path.join(HERE, "Data", "original.csv")

# CDC BRFSS 2015 LLCP variable layout: (start_0, end_0_exclusive, name).
# Derived from the CDC one-column variable layout (Starting Column 1-based
# + Field Length). 326 columns, spanning the full 2155-char record width.
LAYOUT = [
    (0, 2, "_STATE"), (16, 18, "FMONTH"), (18, 26, "IDATE"), (31, 35, "DISPCODE"),
    (35, 45, "SEQNO"), (62, 63, "CTELENUM"), (63, 64, "PVTRESD1"), (64, 65, "COLGHOUS"),
    (65, 66, "STATERES"), (66, 67, "CELLFON3"), (67, 68, "LADULT"), (68, 70, "NUMADULT"),
    (70, 72, "NUMMEN"), (72, 74, "NUMWOMEN"), (74, 75, "CTELNUM1"), (75, 76, "CELLFON2"),
    (76, 77, "CADULT"), (77, 78, "PVTRESD2"), (78, 79, "CCLGHOUS"), (79, 80, "CSTATE"),
    (82, 83, "LANDLINE"), (83, 85, "HHADULT"), (89, 90, "GENHLTH"), (90, 92, "PHYSHLTH"),
    (92, 94, "MENTHLTH"), (94, 96, "POORHLTH"), (96, 97, "HLTHPLN1"), (97, 98, "PERSDOC2"),
    (98, 99, "MEDCOST"), (99, 100, "CHECKUP1"), (100, 101, "BPHIGH4"), (101, 102, "BPMEDS"),
    (102, 103, "BLOODCHO"), (103, 104, "CHOLCHK"), (104, 105, "TOLDHI2"),
    (105, 106, "CVDINFR4"), (106, 107, "CVDCRHD4"), (107, 108, "CVDSTRK3"),
    (108, 109, "ASTHMA3"), (109, 110, "ASTHNOW"), (110, 111, "CHCSCNCR"),
    (111, 112, "CHCOCNCR"), (112, 113, "CHCCOPD1"), (113, 114, "HAVARTH3"),
    (114, 115, "ADDEPEV2"), (115, 116, "CHCKIDNY"), (116, 117, "DIABETE3"),
    (117, 119, "DIABAGE2"), (119, 120, "SEX"), (156, 157, "MARITAL"),
    (157, 158, "EDUCA"), (158, 159, "RENTHOM1"), (167, 168, "NUMHHOL2"),
    (168, 169, "NUMPHON2"), (169, 170, "CPDEMO1"), (170, 171, "VETERAN3"),
    (171, 172, "EMPLOY1"), (172, 174, "CHILDREN"), (174, 176, "INCOME2"),
    (176, 177, "INTERNET"), (177, 181, "WEIGHT2"), (181, 185, "HEIGHT3"),
    (185, 186, "PREGNANT"), (186, 187, "QLACTLM2"), (187, 188, "USEEQUIP"),
    (188, 189, "BLIND"), (189, 190, "DECIDE"), (190, 191, "DIFFWALK"),
    (191, 192, "DIFFDRES"), (192, 193, "DIFFALON"), (193, 194, "SMOKE100"),
    (194, 195, "SMOKDAY2"), (195, 196, "STOPSMK2"), (196, 198, "LASTSMK2"),
    (198, 199, "USENOW3"), (199, 202, "ALCDAY5"), (202, 204, "AVEDRNK2"),
    (204, 206, "DRNK3GE5"), (206, 208, "MAXDRNKS"), (208, 211, "FRUITJU1"),
    (211, 214, "FRUIT1"), (214, 217, "FVBEANS"), (217, 220, "FVGREEN"),
    (220, 223, "FVORANG"), (223, 226, "VEGETAB1"), (226, 227, "EXERANY2"),
    (227, 229, "EXRACT11"), (229, 232, "EXEROFT1"), (232, 235, "EXERHMM1"),
    (235, 237, "EXRACT21"), (237, 240, "EXEROFT2"), (240, 243, "EXERHMM2"),
    (243, 246, "STRENGTH"), (246, 247, "LMTJOIN3"), (247, 248, "ARTHDIS2"),
    (248, 249, "ARTHSOCL"), (249, 251, "JOINPAIN"), (251, 252, "SEATBELT"),
    (252, 253, "FLUSHOT6"), (253, 259, "FLSHTMY2"), (259, 261, "IMFVPLAC"),
    (261, 262, "PNEUVAC3"), (262, 263, "HIVTST6"), (263, 269, "HIVTSTD3"),
    (269, 285, "WHRTST10"), (286, 287, "PDIABTST"), (287, 288, "PREDIAB1"),
    (288, 289, "INSULIN"), (289, 292, "BLDSUGAR"), (292, 295, "FEETCHK2"),
    (295, 297, "DOCTDIAB"), (297, 299, "CHKHEMO3"), (299, 301, "FEETCHK"),
    (301, 302, "EYEEXAM"), (302, 303, "DIABEYE"), (303, 304, "DIABEDU"),
    (304, 306, "PAINACT2"), (306, 308, "QLMENTL2"), (308, 310, "QLSTRES2"),
    (310, 312, "QLHLTH2"), (312, 313, "CAREGIV1"), (313, 315, "CRGVREL1"),
    (315, 316, "CRGVLNG1"), (316, 317, "CRGVHRS1"), (317, 319, "CRGVPRB1"),
    (319, 320, "CRGVPERS"), (320, 321, "CRGVHOUS"), (321, 322, "CRGVMST2"),
    (322, 323, "CRGVEXPT"), (323, 324, "VIDFCLT2"), (324, 325, "VIREDIF3"),
    (325, 326, "VIPRFVS2"), (326, 328, "VINOCRE2"), (328, 329, "VIEYEXM2"),
    (329, 330, "VIINSUR2"), (330, 331, "VICTRCT4"), (331, 332, "VIGLUMA2"),
    (332, 333, "VIMACDG2"), (333, 334, "CIMEMLOS"), (334, 335, "CDHOUSE"),
    (335, 336, "CDASSIST"), (336, 337, "CDHELP"), (337, 338, "CDSOCIAL"),
    (338, 339, "CDDISCUS"), (339, 340, "WTCHSALT"), (340, 343, "LONGWTCH"),
    (343, 344, "DRADVISE"), (344, 346, "ASTHMAGE"), (346, 347, "ASATTACK"),
    (347, 349, "ASERVIST"), (349, 351, "ASDRVIST"), (351, 353, "ASRCHKUP"),
    (353, 356, "ASACTLIM"), (356, 357, "ASYMPTOM"), (357, 358, "ASNOSLEP"),
    (358, 359, "ASTHMED3"), (359, 360, "ASINHALR"), (360, 361, "HAREHAB1"),
    (361, 362, "STREHAB1"), (362, 363, "CVDASPRN"), (363, 364, "ASPUNSAF"),
    (364, 365, "RLIVPAIN"), (365, 366, "RDUCHART"), (366, 367, "RDUCSTRK"),
    (367, 368, "ARTTODAY"), (368, 369, "ARTHWGT"), (369, 370, "ARTHEXER"),
    (370, 371, "ARTHEDU"), (371, 372, "TETANUS"), (372, 373, "HPVADVC2"),
    (373, 375, "HPVADSHT"), (375, 376, "SHINGLE2"), (376, 377, "HADMAM"),
    (377, 378, "HOWLONG"), (378, 379, "HADPAP2"), (379, 380, "LASTPAP2"),
    (380, 381, "HPVTEST"), (381, 382, "HPLSTTST"), (382, 383, "HADHYST2"),
    (383, 384, "PROFEXAM"), (384, 385, "LENGEXAM"), (385, 386, "BLDSTOOL"),
    (386, 387, "LSTBLDS3"), (387, 388, "HADSIGM3"), (388, 389, "HADSGCO1"),
    (389, 390, "LASTSIG3"), (390, 391, "PCPSAAD2"), (391, 392, "PCPSADI1"),
    (392, 393, "PCPSARE1"), (393, 394, "PSATEST1"), (394, 395, "PSATIME"),
    (395, 396, "PCPSARS1"), (396, 397, "PCPSADE1"), (397, 401, "PCDMDECN"),
    (601, 602, "SCNTMNY1"), (602, 603, "SCNTMEL1"), (603, 604, "SCNTPAID"),
    (604, 606, "SCNTWRK1"), (606, 607, "SCNTLPAD"), (607, 609, "SCNTLWK1"),
    (609, 610, "SXORIENT"), (610, 611, "TRNSGNDR"), (617, 618, "RCSGENDR"),
    (654, 655, "RCSRLTN2"), (655, 656, "CASTHDX2"), (656, 657, "CASTHNO2"),
    (657, 658, "EMTSUVRT"), (658, 659, "LSATISFY"), (659, 661, "ADPLEASR"),
    (661, 663, "ADDOWN"), (663, 665, "ADSLEEP"), (665, 667, "ADENERGY"),
    (667, 669, "ADEAT1"), (669, 671, "ADFAIL"), (671, 673, "ADTHINK"),
    (673, 675, "ADMOVE"), (675, 676, "MISTMNT"), (676, 677, "ADANXEV"),
    (681, 683, "QSTVER"), (683, 685, "QSTLANG"), (840, 860, "EXACTOT1"),
    (860, 880, "EXACTOT2"), (1405, 1406, "MSCODE"), (1406, 1412, "_STSTR"),
    (1412, 1422, "_STRWT"), (1442, 1452, "_RAWRAKE"), (1452, 1462, "_WT2RAKE"),
    (1477, 1478, "_CHISPNC"), (1534, 1536, "_CRACE1"), (1536, 1538, "_CPRACE"),
    (1579, 1589, "_CLLCPWT"), (1676, 1677, "_DUALUSE"), (1677, 1687, "_DUALCOR"),
    (1745, 1755, "_LLCPWT"), (1893, 1894, "_RFHLTH"), (1894, 1895, "_HCVU651"),
    (1895, 1896, "_RFHYPE5"), (1896, 1897, "_CHOLCHK"), (1897, 1898, "_RFCHOL"),
    (1898, 1899, "_MICHD"), (1899, 1900, "_LTASTH1"), (1900, 1901, "_CASTHM1"),
    (1901, 1902, "_ASTHMS1"), (1902, 1903, "_DRDXAR1"), (1959, 1961, "_PRACE1"),
    (1961, 1963, "_MRACE1"), (1965, 1966, "_HISPANC"), (1966, 1967, "_RACE"),
    (1967, 1968, "_RACEG21"), (1968, 1969, "_RACEGR3"), (1969, 1970, "_RACE_G1"),
    (1970, 1972, "_AGE5YR"), (1972, 1973, "_AGE65YR"), (1973, 1975, "_AGE80"),
    (1975, 1976, "_AGE_G"), (1976, 1979, "HTIN4"), (1979, 1982, "HTM4"),
    (1982, 1987, "WTKG3"), (1987, 1991, "_BMI5"), (1991, 1992, "_BMI5CAT"),
    (1992, 1993, "_RFBMI5"), (1993, 1994, "_CHLDCNT"), (1994, 1995, "_EDUCAG"),
    (1995, 1996, "_INCOMG"), (1996, 1997, "_SMOKER3"), (1997, 1998, "_RFSMOK3"),
    (1998, 1999, "DRNKANY5"), (1999, 2002, "DROCDY3_"), (2002, 2003, "_RFBING5"),
    (2003, 2008, "_DRNKWEK"), (2008, 2009, "_RFDRHV5"), (2009, 2013, "FTJUDA1_"),
    (2013, 2017, "FRUTDA1_"), (2017, 2021, "BEANDAY_"), (2021, 2025, "GRENDAY_"),
    (2025, 2029, "ORNGDAY_"), (2029, 2033, "VEGEDA1_"), (2033, 2034, "_MISFRTN"),
    (2034, 2035, "_MISVEGN"), (2035, 2036, "_FRTRESP"), (2036, 2037, "_VEGRESP"),
    (2037, 2043, "_FRUTSUM"), (2043, 2049, "_VEGESUM"), (2049, 2050, "_FRTLT1"),
    (2050, 2051, "_VELT1"), (2051, 2052, "_FRT16"), (2052, 2053, "_VEG23"),
    (2053, 2054, "_FRUITEX"), (2054, 2055, "_VEGETEX"), (2057, 2058, "_TOTINDA"),
    (2058, 2061, "METVL11_"), (2061, 2064, "METVL21_"), (2064, 2069, "MAXVO2_"),
    (2069, 2074, "FC60_"), (2074, 2075, "ACTIN11_"), (2075, 2076, "ACTIN21_"),
    (2076, 2079, "PADUR1_"), (2079, 2082, "PADUR2_"), (2082, 2087, "PAFREQ1_"),
    (2087, 2092, "PAFREQ2_"), (2092, 2097, "_MINAC11"), (2097, 2102, "_MINAC21"),
    (2102, 2107, "STRFREQ_"), (2107, 2108, "PAMISS1_"), (2108, 2113, "PAMIN11_"),
    (2113, 2118, "PAMIN21_"), (2118, 2123, "PA1MIN_"), (2123, 2128, "PAVIG11_"),
    (2128, 2133, "PAVIG21_"), (2133, 2138, "PA1VIGM_"), (2138, 2139, "_PACAT1"),
    (2139, 2140, "_PAINDX1"), (2140, 2141, "_PA150R2"), (2141, 2142, "_PA300R2"),
    (2142, 2143, "_PA30021"), (2143, 2144, "_PASTRNG"), (2144, 2145, "_PAREC1"),
    (2145, 2146, "_PASTAE1"), (2146, 2147, "_LMTACT1"), (2147, 2148, "_LMTWRK1"),
    (2148, 2149, "_LMTSCL1"), (2149, 2150, "_RFSEAT2"), (2150, 2151, "_RFSEAT3"),
    (2151, 2152, "_FLSHOT6"), (2152, 2153, "_PNEUMO2"), (2153, 2154, "_AIDTST3"),
]


def convert(input_file, output_file):
    names = [name for _, _, name in LAYOUT]
    colspecs = [(a, b) for a, b, _ in LAYOUT]

    df = pd.read_fwf(
        input_file,
        colspecs=colspecs,
        names=names,
        dtype=str,
        header=None,
        encoding="latin-1",
        skip_blank_lines=True,
    )

    # Strip leading/trailing whitespace (fixed-width padding); blanks -> NA.
    df = df.apply(lambda col: col.str.strip()).replace("", pd.NA)

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    return df.shape


def main():
    args = sys.argv[1:]
    input_file = args[0] if args else DEFAULT_INPUT
    output_file = args[1] if len(args) >= 2 else DEFAULT_OUTPUT

    shape = convert(input_file, output_file)
    print(f"Converted {input_file} -> {output_file}  ({shape[0]} rows x {shape[1]} cols)")


if __name__ == "__main__":
    main()
