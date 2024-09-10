# Msc_Onlab_1
A tanulási képesség mobil környezetben való vizsgálatának elemzése és modellezése.
Név: Porkoláb Mercédesz
A kód magyarázattal Google Codab-ban:
https://colab.research.google.com/drive/1sjea18KFypPhtXN8zGHHk3noS2wKFPOp#scrollTo=kIA3GmU76k9W
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
## Második rész
Be kell tanítani egy neurális hálót, úgy hogy adott egy Mondrián pálya, ahol a pálya 5x5-ös és benne 3 fekete elem található. Egy ilyen pályát kap bemenetként és meg kell állapítania, hogy az adott pálya hány lépésből rakható ki.
## Harmadik rész
Be kell tanítani egy neurális hálót, úgy hogy adott egy Mondrián pálya, ahol a pálya 6x6-os és meg van adva 1-0-kal kirakva, ahol az . Egy ilyen pályát kap bemenetként és meg kell állapítania, hogy az adott pálya hány lépésből rakható ki.
## Negyedik rész
Be kell tanítani egy neurális hálót, úgy hogy adott egy Mondrián pálya, ahol a pálya 6x6-os és meg van adva 1-0-kal kirakva, ahol az . Egy ilyen pályát kap bemenetként és meg kell állapítania, hogy az adott pálya hány lépésből rakható ki, ezt kell a lehető legjobban optimalizálni a hiperparaméterek változatatásával.
## Ötödik rész
A program kap valós paramétereket is tanítóhalmaznak, aminek az első eleme egy 2D-s mátrix, ami a pályát tartalmazza, a második eleme pedig azt hogy hány lépésből kirakható.
## Hatodik rész
* Legyen egy 8x8-as mondrián pályánk, ahol ha kisebb pályára nézzük akkor 1-esekkel fedjük le a részt ahova nem lehet rakni.
* A pályákat forgassuk el minden irányba, ezzel külön eseteket generálva. (Dubla for ciklus)
* Legyen külön tanító, teszt és validációs halmazunk.
* A következő kurzus segít a hiperparaméter választásban (CNN-ig eljutva): https://learn.udacity.com/courses/ud187/welcome-flow
## Hetedik rész
* Osztályzó berakása, könnyű, közepes nehéz osztály, és hogy jó osztályba rakja-e be? --> utolsó réteg lecserélése klasszifikációra
* Csak 8x8-as pálya, és akkor egészítsük ki 1-esekkel ha kevesebb
* Tanítóhalmazba belevinni az összes méretű pályát
* Jupiter notebook-ba belerakni ezeket ".iynb"
* math.lib-bel grafikonokat rajzolni
## Nyolcadik rész
* Random a halmazból 10-et kiveszünk megnézzük arra mit becsül nehézségi szintnek ezt összehasonlítjuk a mostani becslésekkel
* Még több "nehéz" pályát bevenni a halmazba

# Diplomaterv
* Kutatók éjszakája 2024.09.27 - NP teljes probléma - konvolúciós réteg többet lát mint emberi szem, explainable AI , GO játék - AI híres 37. lépes itt is hasonló dolgok lehetnek , pár slide 8-10 perc
* Nemzetkozi AI transformation - okt 25 --> plakátkészítés A0-s / szeptember vége beadással
* TDK - ki kell találnunk a tartalmát, amivel regisztráljuk, dipi terv címe lehet a cím, 2024. okt 1-től lehet regelni.
* Kristóf a Rasch modellel lövi be a feladatok nehézségét. Azt is, amit korábban a Mérő Lacihoz csináltál a modellel. Ezt a módszert le kell lesni tőle, mert próbáljunk olyan modellt is csinálni, ami azt közelíti, IRT kétparaméteres próba - Mondrian blocknál - pszichometrikus - Kristóf
*  HALADNI EZEKKEL: A Mondrian mellett a hexablocks és a háromszöges esetére is kéne hasonló modelleket gyártani. A cogni-teachen is fenn vannak pályák (de már mi is tudjuk, írtunk a nyáron mobil appot, amiben ezek is benne vannak 🙂), A Mondrian mellett a hexablocks és a háromszöges esetére is kéne hasonló modelleket gyártani. A cogni-teachen is fenn vannak pályák (de már mi is tudjuk, írtunk a nyáron mobil appot, amiben ezek is benne vannak 🙂), 2D -s modellé alakítás --> hármszög -- > 2D - A meg B oldal
* kísérletezés a VisualML-lel, mennyire tud ő nehézséget becsülni (de mivel az általános modell, ezért drágábban) --> mostani modell idealakítása - Gyarmati Zsófitól
