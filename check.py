import gdown
import os

def check_pth():
    file_id = '10077FPcef8lrFFDbKfEPzTBPODT3Hys6'

    url = f'https://drive.google.com/uc?id={file_id}'

    if not os.path.exists('./modelo_treinado.pth'):
        print("Para funcionamento da I.A é necessário instalar o arquivo '.pth' de configuração.")
        print("Deseja baixar agora? (Cerca de 132MB serão utilizados)")

        r = input("(s/n): ").lower()

        if r == 's':
            os.system('cls')
            gdown.download(url, './modelo_treinado.pth')
            return True
        else:
            return False