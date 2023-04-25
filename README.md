# Msc_Onlab_1
A tanulási képesség mobil környezetben való vizsgálatának elemzése és modellezése.
Név: Porkoláb Mercédesz
# Hozzáférési adatot a teszteléshez (test userek, backend URL)
Mondrian fiók: felhasználónév: 09dporkolabm@gmail.com
               jelszó: uVAr6kaLhG
# Feladat leírás
A tanulási folyamatok során látens képességek fejlesztése is történik. Korszerű szemléletmód szerint a kognitív képességek tesztelése és fejlesztése helyett vagy mellett játékok és azok eredményének megfigyelése jelenthet szórakoztatóbb, hosszabb távon is érdekes fejlesztési módot.

A téma kapcsán lehetőség van különböző aspektusokból bekapcsolódni ebbe a kutatásba.

Játékfejlesztéssel: a kutatócsoport tagjai (mint pl. Mérő László, Kökényesi Imre, Gyarmathy Éva) által kidolgozott (vagy prototípus szintjén elérhető) játékok digitalizálása Android környezetbe
Pszichometriai elemző modulok fejlesztésével (látens képesség kinyerése célzott játékok, tesztek eredményének statisztikai apparátussal történő elemzésével)
Adaptív mérőskálák kidolgozásával (A sakkjátékokból is ismert Élő - módszerrel, vagy Urnings módszerrel, vagy egyéb, a feleletválasztós tesztek mögötti statisztikai modellek paramétereit mérések
 
A fenti eljárásokhoz a kutatási terület iránt is érdeklődő hallgató az érdeklődésének megfelelő apparátust választhat (pl. statisztikai módszertanok, neurális hálók, Bayes-i döntések, újszerű modellek kidolgozása)
# Egyedi feladathoz a kutatás és források
Mondrian játéhoz link: www.mondrianblocks.us
Kognitív profil teszt:  https://kognitivprofil.hu/ 

Kutatás forrásai: https://onedrive.live.com/?authkey=%21AEz7N6kMR%2Dgg42s&id=DEC322D864B6AF18%21205554&cid=DEC322D864B6AF18

* Rasch könyv: 5-11. fejezetek
* McGrew cikk egy state-of-the-art képesség felosztásról
* Maria Bolsinova: az Urnings modellről

Felhasználandó program a Mondrián játék kimeneteléhez: https://github.com/cyberci/MondrianSimulator

Tensorflow Codelabs: https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist#4


# Részletes specifikáció, haladással
## Első rész
Be kell tanítani egy neurális hálót, úgy hogy adott egy Mondrián pálya, ahol a pálya 5x5-ös és benne 3 fekete elem található. Egy ilyen pályát kap bemenetként és meg kell állapítania, hogy az adott pálya milyen nehézségű. (Százalékos formában.)
##Második rész
Be kell tanítani egy neurális hálót, úgy hogy adott egy Mondrián pálya, ahol a pálya 5x5-ös és benne 3 fekete elem található. Egy ilyen pályát kap bemenetként és meg kell állapítania, hogy az adott pálya hány lépésből rakható ki.
##Harmadik rész
Be kell tanítani egy neurális hálót, úgy hogy adott egy Mondrián pálya, ahol a pálya 6x6-os és meg van adva 1-0-kal kirakva, ahol az . Egy ilyen pályát kap bemenetként és meg kell állapítania, hogy az adott pálya hány lépésből rakható ki.
##Negyedik rész
Be kell tanítani egy neurális hálót, úgy hogy adott egy Mondrián pálya, ahol a pálya 6x6-os és meg van adva 1-0-kal kirakva, ahol az . Egy ilyen pályát kap bemenetként és meg kell állapítania, hogy az adott pálya hány lépésből rakható ki, ezt kell a lehető legjobban optimalizálni a hiperparaméterek változatatásával.
##Ötödik rész
A program kap valós paramétereket is tanítóhalmaznak, aminek az első eleme egy 2D-s mátrix, ami a pályát tartalmazza, a második eleme pedig azt hogy hány lépésből kirakható.
##Hatodik rész
Legyen egy 8x8-as mondrián pályánk, ahol ha kisebb pályára nézzük akkor 1-esekkel fedjük le a részt ahova nem lehet rakni.
A pályákat forgassuk el minden irányba, ezzel külön eseteket generálva. (Dubla for ciklus)
Legyen külön tanító, teszt és validációs halamzunk.
A következő kurzus segít a hiperparaméter választásban (CNN-ig eljutva): https://learn.udacity.com/courses/ud187/welcome-flow

