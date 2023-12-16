import check
import PySimpleGUI as sg
from PIL import Image, ImageTk
import os

def main():
    
    layout = [
        [sg.Image(key='-IMAGE-', size=(300, 300))],
        [sg.Text(key='-lblClass-')],
        [sg.Text(key='-lblProb-')],        
        [sg.Button('Abrir Imagem', size=(20, 1)), sg.Button('Classificar', size=(20, 1))]
    ]

    window = sg.Window('Classificadora de Imagens', layout, element_justification='c', resizable=True, finalize=True)
    filename = ""

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        elif event == 'Abrir Imagem':
            filename = sg.popup_get_file('Selecione uma imagem', file_types=(("Imagens", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")))

            if filename:
                pil_image = Image.open(filename).resize((256, 256))
                tk_image = ImageTk.PhotoImage(pil_image)

                window['-lblClass-'].update(f'')
                window['-lblProb-'].update(f'')
                window['-IMAGE-'].update(data=tk_image)

        elif event == "Classificar":
            classe, prob = model.classificar_imagem(filename)
            window['-lblClass-'].update(f'Classificada como: {classe}')
            window['-lblProb-'].update(f'Probabilidade: {prob}%')

    window.close()

if __name__ == '__main__':
    if not check.check_pth():
        exit()
    os.system('cls')

    print("Carregando modelo de I.A...")
    import model
    os.system('cls')

    main()

