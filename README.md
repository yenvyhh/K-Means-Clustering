# Projekt 3 - K Means Clustering
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yenvyhh/K-Means-Clustering/main?filepath=K%20Means%20Clustering%20-%20Projekt%203.ipynb)

**Zu Beginn bitte unter "Cell" -> "Run All" auswählen.**

**Die Daten importieren,als DataFrame abspeichern und das Head anzeigen lassen:**
df = pd.read_csv("College_Data",index_col=0) 
df.head()
**Nach Ausführung sollten von der importierten Datei die ersten 5 Zeilen angezeigt werden. Die In den Zeilen sind die jeweiligen Namen der Universitäten eingetragen** 
       
**Informationen und Details des Data Frames bzw. der Daten anzeigen lassen:**     
df.info()
df.describe()
**Bei Info wird angezeigt, ob die Spalten einen Float, ein Integer oder ein Object sind. Zu dem wird bei RangeIndex angezeigt, dass es 777 Einträge gibt. Bei Describe wird ein Dataset der Analyse geprintet. Beispiele hierfür sind der Durchschnittswert, der Minimum- oder Maximum-Wert.

**Darauffolgend erfolgt eine EXPLORATIVE DATENANALYSE, die durch verschiedene Diagrammvisualisierungen dargestellt werden. Ein Beispiel, das ausgeführt wird:**
sns.set_style("whitegrid")
sns.scatterplot(x="Room.Board",
    y="Grad.Rate",
    hue="Private",
    data=df,
    palette="coolwarm")
**Durch Ausführen der ganzen Befehle werden Scatterplot (Streuiagramm) erstellt, wo nach der Kategorie Privat ("Ja") oder Öffentlich ("Nein) unterschieden wird. Die x-Achse ist die Room.Board (Kosten für Räume und Mitarbeiter) und die y-Achse die Grad.Rate (Abschlussrate). Das selbe Diagramm lässt sich auch wie folgt darstellen:**
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',
           palette='coolwarm',height=6,aspect=1,fit_reg=False)

**Ein weiteres Beispiel:**
sns.set_style("darkgrid")
plt.figure(figsize=(12,6))
df[df["Private"]=="Yes"]["Grad.Rate"].hist(alpha=0.4,bins=20)
df[df["Private"]=="No"]["Grad.Rate"].hist(color="red",alpha=0.4,bins=20)
plt.xlabel("Grad.Rate")
**Dieses Histogramm zeigt die Abschussrate für öffentliche und private Universitäten. Dabei fällt auf, dass eine Universität eine Abschlussrate > 100 % hat. Den Namen der Universtiät erhält man wie folgt:**
df[df["Grad.Rate"] >100]

**Im nächsten Schritt wird das K Means Cluster erstellt:**
kmeans = KMeans(n_clusters=2)

**Nach Erstellung der K Means Modell mit 2 Clustern wird das Modell gefittet:**
kmeans.fit(df.drop("Private",axis=1))
kmeans.cluster_centers_
**Jetzt sollten die Cluster Zentrumsvektoren im Array angezeigt werden.


**Zur Auswertung müssen die kategorischen Merkmale in 0 (öffentlich) und 1 (privat umgewandelt werden und in eine neue Spalte eingefügt werden.**
def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0
 
df["Cluster"] = df["Private"].apply(converter)
df.head()

**Nun sollte die Liste mit der neuen Spalte "Cluster" ergänzt sein.

**Basierend darauf kann ein Klassifizierungsreport und eine Confusion Matrix für das Modell erstellt werden:**
print (confusion_matrix(df["Cluster"],kmeans.labels_))
print ("\n")
print (classification_report(df["Cluster"],kmeans.labels_))
**Je näher die Werte bei precicion, recall und f1-score an 1 sind, desto genauer sind Auswertung. **

