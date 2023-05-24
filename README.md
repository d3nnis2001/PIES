# PIES der beste Algo an der Wallstreet

## Downloads
---
Ihr braucht eine IDE. Am besten downloaded ihr VSCode falls ihr das noch nicht habt: 
- https://code.visualstudio.com/

Falls ihr bei Mac von zsh auf bash wechseln wollt/müsst gibt folgenden Befehl ein:

- chsh -s /bin/bash

Downloaded die 3.8 Version von Python. Wichtig dabei ist beim Installierungsprozess den Kasten anzudrücken wo steht: Add to PATH oder add to environment
- https://www.python.org/downloads/release/python-380/

Hier könnt ihr mit: 
- python3 --version
in der Konsole überprüfen ob alles gut verlief

Downloaded Miniconda. Hier müsst ihr wieder den zum PATH hinzufügen. Beachtet das es zwei Versionen gibt bei Macbooks. Einfach für die mit den M1 und M2 chips und für die Intel Prozessoren 
- https://docs.conda.io/en/latest/miniconda.html

Um zu überprüfen ob Miniconda richtig installiert wurde: 

- conda --version

Falls ihr noch kein homebrew habt müsst ihr es erst installieren mit der Bash Konsole: 
- /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" 

Falls ihr über Mac seit müsst ihr git installieren. Bei windows ist es bereits normalerweise drauf. Dafür müsst ihr nur im Terminal __folgendes__ eingeben: 
- brew install git

Um zu überprüfen ob git richtig installiert wurde: 

- git --version

---

Jetzt steht das Grundgerüst. Jetzt müssen wir innerhalb von VSCode arbeiten.

Erstmal erstellen wir einen neuen Ordner und öffnen diesen oben rechts bei Datei. So jetzt müssen wir folgende Extensions downloaden:

- Jupyter

Jetzt müssen wir unser git mit Github verbinden. Dafür müsst ihr folgendes in die Konsole eingeben: 

- git config --global user.name "Your name here"
- git config --global user.email "your_email@example.com"
 
So jetzt könnt ihr die Repository clonen:

- git clone "https://github.com/d3nnis2001/Phillips_Algo.git"

Jetzt haben wir endlich das Projekt lokal gespeichert. Nun will ich das ihr euren eigenen Branch kreiert. Das ist sogesagt euer eigener Zweig wo ihr machen könnt was ihr wollt. 

- git checkout -b BranchNamenKoenntIhrEuchAussuchen

Damit wechselt ihr in den Zweig und kreiert ihn. 

Falls irgendwelche Änderung da sind könnt ihr folgendes machen. 

Um für Änderungen zu checken:

- git status

falls Änderung da sind: 

- git pull

falls ihr irgendwelche changes gemacht habt macht folgendes um git pull ausführen zu können: 

- git add .
- git commit -m "Hier die Nachricht was ihr geändert habt"
- git push origin ZweigName

So das sind alle git basics.

Jetzt müssen wir die Libaries installieren, damit ihr den code laufen lassen könnt.

- conda env create -f environment.yaml

Damit installiert ihr alle libaries und jetzt müsst ihr mit folgenden Befehl die environment aktivieren:

- conda activate pie

Jetzt müsst ihr nurnoch den richtigen Kernel auswählen (pie) und schon sind wir fertig.

git status

Falls Änderung gemacht wurden:

git add .
git commit -m "Hab bisschen rumprobiert"
git push 

Changes zu euch holen:

git pull origin main

git branch 