class Style:
    CYBORG = 'goofyai/cyborg_style_xl'
    PIXEL_ART = 'kohbanye/pixel-art-style'
    REALISTIC = 'SG161222/Realistic_Vision_V1.4'
    DARK_DARKER = 'ai-characters/DarkAndDarker-Style-SDXL'
    COMICS = 'VuDucQuang/comics-character-style'
    HERGE = 'sd-dreambooth-library/herge-style'
    PENCIL_DRAW = 'ProomptEngineer/pe-pencil-drawing-style'
    DEFAULT = 'DEFAULT'
    

def StyleFrom(pos) -> str:
        match pos:
            case 1: 
                return Style.CYBORG
            case 2: 
                return Style.PIXEL_ART
            case 3: 
                return Style.REALISTIC
            case 4: 
                return Style.DARK_DARKER
            case 5: 
                return Style.COMICS
            case 6: 
                return Style.HERGE
            case 7: 
                return Style.PENCIL_DRAW
            case _:
                return Style.DEFAULT    