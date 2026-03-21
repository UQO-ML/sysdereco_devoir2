# Analyse qualitative 3.2.3
Variante: active_pre_split
Utilisateurs analyses: 5

================================================================================
U1_gros_lecteur | user_id=AH2GI5KLJT2AKVCYXV2OIIGGLT4Q_1
Selection: nb_train >= q95 (99.0)

[Historique train]
```
                                                                       titre                                                  categories  rating          date
                                                             Mercy (A Novel)  Books | Politics & Social Sciences | Politics & Government     3.0 1232656934000
                                            13 1/2 Lives of Captain Bluebear               Books | Literature & Fiction | Humor & Satire     4.0 1232657754000
                                                     Family History: A Novel                Books | Literature & Fiction | Genre Fiction     5.0 1232658304000
                                               Island of Lost Girls: A Novel Books | Mystery, Thriller & Suspense | Thrillers & Suspense     4.0 1232658387000
                                         The Amber Room: A Novel of Suspense           Books | Literature & Fiction | Action & Adventure     4.0 1232659948000
                                   Breaking Dawn (The Twilight Saga, Book 4)      Books | Teen & Young Adult | Science Fiction & Fantasy     3.0 1232660487000
                                                   The Tenth Circle: A Novel  Books | Politics & Social Sciences | Politics & Government     2.0 1232660575000
                  The Story of Edgar Sawtelle: A Novel (Oprah Book Club #62)                Books | Literature & Fiction | Genre Fiction     5.0 1232660877000
                                                      The Pact: A Love Story                Books | Literature & Fiction | Genre Fiction     5.0 1232660953000
                                    Stolen (Women of the Otherworld, Book 2)                Books | Literature & Fiction | Genre Fiction     5.0 1232661255000
                                                                    The Loch Books | Mystery, Thriller & Suspense | Thrillers & Suspense     5.0 1232661457000
                              Uncle Tungsten: Memories of a Chemical Boyhood   Books | Biographies & Memoirs | Professionals & Academics     2.0 1234199014000
                                                                  Back Roads                Books | Literature & Fiction | Genre Fiction     5.0 1234202230000
                                                     One True Thing: A Novel                Books | Literature & Fiction | Genre Fiction     5.0 1234452639000
            The Woman Who Walked into Doors: A Novel (A Paula Spencer Novel)                 Books | Literature & Fiction | Contemporary     5.0 1235056813000
Little Face: A Zailer and Waterhouse Mystery (A Zailer & Waterhouse Mystery) Books | Mystery, Thriller & Suspense | Thrillers & Suspense     4.0 1235489349000
                                    Poison Study (The Chronicles of Ixia, 1)                 Books | Science Fiction & Fantasy | Fantasy     5.0 1235666576000
                                                                 Magic Study                 Books | Science Fiction & Fantasy | Fantasy     5.0 1235670831000
                                                  Fire Study (Study, Book 3)                 Books | Science Fiction & Fantasy | Fantasy     5.0 1235749102000
                                                 Peace Like a River: A Novel                Books | Literature & Fiction | Genre Fiction     5.0 1236017951000
```

[Recommandations top N]
```
 rang       asin                                                               titre_recommande                                           categories                score_source
    1 B003NHR9EM                                                             The Espressologist                                                      top_n_indices_similarity.py
    2 0399161287                                                               Undisputed Truth              Books | Sports & Outdoors | Biographies top_n_indices_similarity.py
    3 0671025368                                                        The Coldest Winter Ever         Books | Literature & Fiction | United States top_n_indices_similarity.py
    4 0765326361 Words of Radiance (The Stormlight Archive, Book 2) (The Stormlight Archive, 2)    Books | Literature & Fiction | Action & Adventure top_n_indices_similarity.py
    5 0671725823                                                       Miles: The Autobiography Books | Politics & Social Sciences | Social Sciences top_n_indices_similarity.py
    6 1501110365                                                   It Ends with Us: A Novel (1)       Books | Literature & Fiction | Women's Fiction top_n_indices_similarity.py
    7 0143124544                                          Me Before You (Me Before You Trilogy)         Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
    8 0670026603                                 Me Before You: A Novel (Me Before You Trilogy)         Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
    9 B0090UOJAI                                         Cold Days (The Dresden Files, Book 14)       Books | Mystery, Thriller & Suspense | Mystery top_n_indices_similarity.py
   10 0316187291                                           The Shadowed Sun (The Dreamblood, 2)         Books | Literature & Fiction | United States top_n_indices_similarity.py
```

[Verite terrain test]
```
                                                         titre                                                  categories  rating
                                           Providence: A Novel                Books | Literature & Fiction | Genre Fiction     4.0
                               Something in the Water: A Novel Books | Mystery, Thriller & Suspense | Thrillers & Suspense     4.0
The Girl Who Smiled Beads: A Story of War and What Comes After         Books | Biographies & Memoirs | Community & Culture     4.0
                                                   This I Know                Books | Literature & Fiction | Genre Fiction     5.0
                                           The Pisces: A Novel                Books | Literature & Fiction | Genre Fiction     3.0
                                                  Ash Princess      Books | Teen & Young Adult | Science Fiction & Fantasy     5.0
                                 The Flight Attendant: A Novel Books | Mystery, Thriller & Suspense | Thrillers & Suspense     4.0
                                               Neverworld Wake           Books | Teen & Young Adult | Literature & Fiction     5.0
                   Foundryside: A Novel (The Founders Trilogy)           Books | Literature & Fiction | Action & Adventure     5.0
                                         Heartbreaker: A Novel                Books | Literature & Fiction | Genre Fiction     1.0
                                       Where the Crawdads Sing                Books | Literature & Fiction | Genre Fiction     3.0
                                               The War Outside           Books | Teen & Young Adult | Literature & Fiction     5.0
                                                 Grim Lovelies      Books | Teen & Young Adult | Science Fiction & Fantasy     5.0
                                       #FashionVictim: A Novel           Books | Crafts, Hobbies & Home | Crafts & Hobbies     4.0
                            Twice Dead: The Necromancer's Song      Books | Teen & Young Adult | Science Fiction & Fantasy     4.0
                                     A Spark of Light: A Novel                Books | Literature & Fiction | Genre Fiction     2.0
                                        Pieces of Her: A Novel                Books | Literature & Fiction | Genre Fiction     4.0
                                    We Sold Our Souls: A Novel                Books | Literature & Fiction | Genre Fiction     4.0
                                       Sister of Mine: A Novel           Books | Teen & Young Adult | Literature & Fiction     3.0
                                    The Drama Teacher: A Novel                Books | Literature & Fiction | Genre Fiction     5.0
```

[Analyse 4 axes]
- Coherence thematique: recouvrement categories = 0.14
- Proximite apparente (exemples):
  * 'The Loch' -> 'The Espressologist' (sim=0.65)
  * 'The Loch' -> 'Miles: The Autobiography' (sim=0.46)
  * 'The Loch' -> 'Words of Radiance (The Stormlight Archive, Book 2) (The Stormlight Archive, 2)' (sim=0.45)
- Redondance: similarite intra liste moyenne = 0.64, max = 0.99
- Sur specialisation: concentration categorie dominante train = 0.20
- Hit rate test: 0/10
- Top categories train: Books | Literature & Fiction | Genre Fiction (27.4%), Books | Mystery (23.1%), Thriller & Suspense | Thrillers & Suspense (15.9%)
- Top categories recos: Books | Literature & Fiction | United States (20.0%), Books | Literature & Fiction | Genre Fiction (20.0%), Books | Literature & Fiction | Action & Adventure (10.0%)

================================================================================
U2_lecteur_modere | user_id=AH5GNK4DJYJLZDIEDV6EF5JRA4EQ
Selection: nb_train proche mediane (24.0)

[Historique train]
```
                                                                           titre                                                 categories  rating          date
                                                                     Wolf Hollow            Books | Children's Books | Literature & Fiction     5.0 1484172775000
                                          Some Writer!: The Story of E. B. White           Books | Children's Books | Education & Reference     5.0 1484177977000
                                             Charlotte the Scientist Is Squished      Books | Children's Books | Growing Up & Facts of Life     5.0 1489630264000
                                               7 Ate 9 (Volume 1) (Private I, 1)                  Books | Children's Books | Early Learning     5.0 1498421740798
                                    Water Is Water: A Book About the Water Cycle  Books | Children's Books | Science, Nature & How It Works     5.0 1499200590470
                                                                    Jabari Jumps      Books | Children's Books | Growing Up & Facts of Life     5.0 1499200689412
                               Not Quite Narwhal (Not Quite Narwhal and Friends)      Books | Children's Books | Growing Up & Facts of Life     5.0 1500577971698
                                                          The First Rule of Punk       Books | Children's Books | Arts, Music & Photography     5.0 1504654582535
                                          The Hate U Give: A Printz Honor Winner          Books | Teen & Young Adult | Literature & Fiction     5.0 1506446594812
                                                       Real Friends (Friends, 1)      Books | Children's Books | Growing Up & Facts of Life     5.0 1509161351952
                            After the Fall (How Humpty Dumpty Got Back Up Again) Books | Children's Books | Fairy Tales, Folk Tales & Myths     5.0 1509324508295
                                                                     Dear Martin          Books | Teen & Young Adult | Literature & Fiction     5.0 1510010244764
                        The Vanderbeekers of 141st Street (The Vanderbeekers, 1)      Books | Children's Books | Growing Up & Facts of Life     5.0 1510325688025
                         Insignificant Events in the Life of a Cactus (Volume 1)      Books | Children's Books | Growing Up & Facts of Life     5.0 1513539390948
The 57 Bus: A True Story of Two Teenagers and the Crime That Changed Their Lives Books | Politics & Social Sciences | Politics & Government     5.0 1513896896149
                          She Persisted: 13 American Women Who Changed the World      Books | Children's Books | Growing Up & Facts of Life     5.0 1514435732894
                                              The Rooster Who Would Not Be Quiet      Books | Children's Books | Growing Up & Facts of Life     5.0 1516219184035
                                                    Love, Hate and Other Filters          Books | Teen & Young Adult | Literature & Fiction     5.0 1517540087932
                                                                            Life  Books | Children's Books | Science, Nature & How It Works     5.0 1520207830747
                                                                  American Panda          Books | Teen & Young Adult | Literature & Fiction     5.0 1520780041648
```

[Recommandations top N]
```
 rang       asin                                                                        titre_recommande                                                categories                score_source
    1 B003NHR9EM                                                                      The Espressologist                                                           top_n_indices_similarity.py
    2 1338180614                                                                Baby Monkey, Private Eye           Books | Children's Books | Literature & Fiction top_n_indices_similarity.py
    3 0316324906 Finding Winnie: The True Story of the World's Most Famous Bear (Caldecott Medal Winner)           Books | Children's Books | Literature & Fiction top_n_indices_similarity.py
    4 1909263990                                                                             The Journey     Books | Children's Books | Growing Up & Facts of Life top_n_indices_similarity.py
    5 1328780961                                                   The Undefeated (Caldecott Medal Book)           Books | Children's Books | Geography & Cultures top_n_indices_similarity.py
    6 0062996487                                                                        Punching the Air         Books | Teen & Young Adult | Literature & Fiction top_n_indices_similarity.py
    7 1632173182                Library Girl: How Nancy Pearl Became America's Most Celebrated Librarian     Books | Children's Books | Growing Up & Facts of Life top_n_indices_similarity.py
    8 1596439645                                                            School's First Day of School     Books | Children's Books | Growing Up & Facts of Life top_n_indices_similarity.py
    9 0316381993                                                      The Wild Robot (The Wild Robot, 1) Books | Children's Books | Science, Nature & How It Works top_n_indices_similarity.py
   10 0399257748                                                              Last Stop on Market Street     Books | Children's Books | Growing Up & Facts of Life top_n_indices_similarity.py
```

[Verite terrain test]
```
                                                                           titre                                            categories  rating
Girl, Stop Apologizing: A Shame-Free Plan for Embracing and Achieving Your Goals           Books | Business & Money | Business Culture     2.0
                                                                    Roll with It Books | Children's Books | Growing Up & Facts of Life     4.0
                                                 From the Desk of Zoe Washington Books | Children's Books | Growing Up & Facts of Life     5.0
                                                                           Furia     Books | Teen & Young Adult | Literature & Fiction     5.0
                          Maizy Chen's Last Chance: (Newbery Honor Award Winner) Books | Children's Books | Growing Up & Facts of Life     5.0
```

[Analyse 4 axes]
- Coherence thematique: recouvrement categories = 0.36
- Proximite apparente (exemples):
  * 'Wolf Hollow' -> 'The Espressologist' (sim=0.67)
  * 'Wolf Hollow' -> 'The Journey' (sim=0.54)
  * 'Wolf Hollow' -> 'Finding Winnie: The True Story of the World's Most Famous Bear (Caldecott Medal Winner)' (sim=0.53)
- Redondance: similarite intra liste moyenne = 0.59, max = 0.71
- Sur specialisation: concentration categorie dominante train = 0.40
- Hit rate test: 0/10
- Top categories train: Books | Children's Books | Growing Up & Facts of Life (32.1%), Books | Teen & Young Adult | Literature & Fiction (17.9%), Nature & How It Works (7.1%)
- Top categories recos: Books | Children's Books | Growing Up & Facts of Life (40.0%), Books | Children's Books | Literature & Fiction (20.0%), Books | Children's Books | Geography & Cultures (10.0%)

================================================================================
U3_petit_lecteur | user_id=AE3RDSKZEG2MRMMY32NZMEDO3XEQ
Selection: nb_train 3-5 (ou minimum disponible)

[Historique train]
```
                                                                                                                     titre                                      categories  rating          date
Knights and Castles: A Nonfiction Companion to Magic Tree House #2: The Knight at Dawn (Magic Tree House (R) Fact Tracker) Books | Children's Books | Literature & Fiction     5.0  966006340000
                                                                                                         Goldman's Theorem                    Books | Literature & Fiction     5.0 1238958503000
                                                                    The Night Before Christmas Recordable Story (Hallmark)                                                     5.0 1259802495000
```

[Recommandations top N]
```
 rang       asin                                                                                                                titre_recommande                                              categories                score_source
    1 B003NHR9EM                                                                                                              The Espressologist                                                         top_n_indices_similarity.py
    2 037585651X                                                                 Leprechaun in Late Winter (Magic Tree House (R) Merlin Mission)         Books | Children's Books | Literature & Fiction top_n_indices_similarity.py
    3 0800739337                                                                                    A Christmas in the Alps: A Christmas Novella Books | Christian Books & Bibles | Literature & Fiction top_n_indices_similarity.py
    4 0375813659 Magic Tree House Boxed Set, Books 1-4: Dinosaurs Before Dark, The Knight at Dawn, Mummies in the Morning, and Pirates Past Noon                   Books | Boxed Sets | Children's Books top_n_indices_similarity.py
    5 0778386562                                                                            The Road to Christmas: A Sweet Holiday Romance Novel            Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
    6 1250164672                                                                                   The Christmas Table: A Novel (Christmas Hope)            Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
    7 1947892371                                                                                                           The Secret Ingredient            Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
    8 0800736109                                                                                                        Christmas in Winter Hill Books | Christian Books & Bibles | Literature & Fiction top_n_indices_similarity.py
    9 077838635X                                                                         Sand Dollar Lane: A Novel (A Moonlight Harbor Novel, 6)            Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
   10 1947892991                                                       Christmas Charms: A small-town Christmas romance from Hallmark Publishing            Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
```

[Verite terrain test]
```
                          titre                                       categories  rating
Nikon D5000 Digital Field Guide Books | Arts & Photography | Photography & Video     1.0
```

[Analyse 4 axes]
- Coherence thematique: recouvrement categories = 0.50
- Proximite apparente (exemples):
  * 'Knights and Castles: A Nonfiction Companion to Magic Tree House #2: The Knight at Dawn (Magic Tree House (R) Fact Tracker)' -> 'Leprechaun in Late Winter (Magic Tree House (R) Merlin Mission)' (sim=0.82)
  * 'Knights and Castles: A Nonfiction Companion to Magic Tree House #2: The Knight at Dawn (Magic Tree House (R) Fact Tracker)' -> 'Magic Tree House Boxed Set, Books 1-4: Dinosaurs Before Dark, The Knight at Dawn, Mummies in the Morning, and Pirates Past Noon' (sim=0.70)
  * 'Knights and Castles: A Nonfiction Companion to Magic Tree House #2: The Knight at Dawn (Magic Tree House (R) Fact Tracker)' -> 'The Espressologist' (sim=0.69)
- Redondance: similarite intra liste moyenne = 0.61, max = 0.95
- Sur specialisation: concentration categorie dominante train = 0.11
- Hit rate test: 0/10
- Top categories train: Books | Children's Books | Literature & Fiction (50.0%), Books | Literature & Fiction (50.0%)
- Top categories recos: Books | Literature & Fiction | Genre Fiction (55.6%), Books | Christian Books & Bibles | Literature & Fiction (22.2%), Books | Children's Books | Literature & Fiction (11.1%)

================================================================================
U4_lecteur_eclectique | user_id=AH665SQ6SQF6DXAGYIQFCX76LALA
Selection: n_categories >= q95 (24.0)

[Historique train]
```
                                                                    titre                                                         categories  rating          date
The Torment of Others: A Novel (Dr. Tony Hill and Carol Jordan Mysteries)                     Books | Mystery, Thriller & Suspense | Mystery     3.0 1116963906000
           The Twelfth Card: A Lincoln Rhyme Novel (Lincoln Rhyme Novels)                       Books | Literature & Fiction | United States     3.0 1118696031000
                                                           Dance of Death        Books | Mystery, Thriller & Suspense | Thrillers & Suspense     3.0 1125005119000
                                                                   Vanish        Books | Mystery, Thriller & Suspense | Thrillers & Suspense     4.0 1132426280000
                                       Predator (Kay Scarpetta Mysteries)        Books | Mystery, Thriller & Suspense | Thrillers & Suspense     2.0 1132698474000
                                                               Mary, Mary        Books | Mystery, Thriller & Suspense | Thrillers & Suspense     3.0 1137353514000
                                   Transgressions: Ten Brand-New Novellas                       Books | Literature & Fiction | United States     4.0 1139442439000
                         The Sleeping Doll: A Novel (Kathryn Dance, No 1)        Books | Mystery, Thriller & Suspense | Thrillers & Suspense     3.0 1213055268000
                                                          One Last Scream                     Books | Mystery, Thriller & Suspense | Mystery     4.0 1222810473000
                                 Devil Bones (A Temperance Brennan Novel)        Books | Mystery, Thriller & Suspense | Thrillers & Suspense     3.0 1246215443000
                                                            I Can See You                       Books | Literature & Fiction | United States     3.0 1250219414000
                                           Triptych: A Novel (Will Trent)                     Books | Mystery, Thriller & Suspense | Mystery     4.0 1251668525000
                                  Roadside Crosses: A Kathryn Dance Novel                     Books | Mystery, Thriller & Suspense | Mystery     3.0 1252095534000
                        The Lovers: A Thriller (Charlie Parker Thrillers)                     Books | Mystery, Thriller & Suspense | Mystery     4.0 1254077220000
                                                           Cemetery Dance        Books | Mystery, Thriller & Suspense | Thrillers & Suspense     3.0 1265512428000
  Scent of the Missing: Love and Partnership With a Search-and-Rescue Dog                Books | Crafts, Hobbies & Home | Pets & Animal Care     5.0 1268194115000
                                   The Cold Room (A Taylor Jackson Novel)        Books | Mystery, Thriller & Suspense | Thrillers & Suspense     3.0 1271000221000
       Create Your Own Blog: 6 Easy Projects to Start Blogging Like a Pro           Books | Computers & Technology | Internet & Social Media     4.0 1273351010000
                                                                  Vicious        Books | Mystery, Thriller & Suspense | Thrillers & Suspense     4.0 1276624575000
      The Women's Small Business Start-Up Kit: A Step-by-Step Legal Guide Books | Business & Money | Business Development & Entrepreneurship     4.0 1277325430000
```

[Recommandations top N]
```
 rang       asin                                                               titre_recommande                                           categories                score_source
    1 B003NHR9EM                                                             The Espressologist                                                      top_n_indices_similarity.py
    2 0399161287                                                               Undisputed Truth              Books | Sports & Outdoors | Biographies top_n_indices_similarity.py
    3 0735290512         Daring to Hope: Finding God's Goodness in the Broken and the Beautiful  Books | Christian Books & Bibles | Christian Living top_n_indices_similarity.py
    4 0671025368                                                        The Coldest Winter Ever         Books | Literature & Fiction | United States top_n_indices_similarity.py
    5 034551100X    A Mighty Long Way: My Journey to Justice at Little Rock Central High School                           Books | History | Americas top_n_indices_similarity.py
    6 0671725823                                                       Miles: The Autobiography Books | Politics & Social Sciences | Social Sciences top_n_indices_similarity.py
    7 1328466892                                       The Thief Knot: A Greenglass House Story Books | Children's Books | Science Fiction & Fantasy top_n_indices_similarity.py
    8 0765326361 Words of Radiance (The Stormlight Archive, Book 2) (The Stormlight Archive, 2)    Books | Literature & Fiction | Action & Adventure top_n_indices_similarity.py
    9 0143124544                                          Me Before You (Me Before You Trilogy)         Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
   10 0670026603                                 Me Before You: A Novel (Me Before You Trilogy)         Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
```

[Verite terrain test]
```
                                                                                              titre                                                    categories  rating
                                                                                   Time Is a Killer   Books | Mystery, Thriller & Suspense | Thrillers & Suspense     3.0
                                                                             Half Moon Bay: A Novel   Books | Mystery, Thriller & Suspense | Thrillers & Suspense     2.0
                               Beauty in the Broken Places: A Memoir of Love, Faith, and Resilience     Books | Biographies & Memoirs | Professionals & Academics     3.0
                                                                          The Family Tabor: A Novel                  Books | Literature & Fiction | Genre Fiction     2.0
                                                                                The Pisces: A Novel                  Books | Literature & Fiction | Genre Fiction     3.0
                                               Ready to Go Guided Reading: Synthesize, Grades 1 - 2             Books | Education & Teaching | Schools & Teaching     4.0
                                                       Strongheart: Wonder Dog of the Silver Screen               Books | Children's Books | Literature & Fiction     5.0
                                                                                         The Window             Books | Teen & Young Adult | Literature & Fiction     4.0
                                                                      Rattle (The Collector Series)   Books | Mystery, Thriller & Suspense | Thrillers & Suspense     4.0
                                                                       Our Kind of Cruelty: A Novel                  Books | Literature & Fiction | Genre Fiction     4.0
                                                                                       The Neighbor                  Books | Literature & Fiction | Genre Fiction     2.0
                                                                                    American By Day             Books | Literature & Fiction | Action & Adventure     3.0
                                                                  Celebrate Learning Shape Stickers         Books | Children's Books | Activities, Crafts & Games     5.0
                                                        Kicks: The Great American Story of Sneakers                         Books | Business & Money | Industries     4.0
                                                                         Forsaken (A Unit 51 Novel)                  Books | Literature & Fiction | Genre Fiction     2.0
                                                    Chernobyl: The History of a Nuclear Catastrophe                                    Books | History | Military     4.0
                                                                                  The Banker's Wife                  Books | Literature & Fiction | Genre Fiction     3.0
Grow Something Different to Eat: Weird and wonderful heirloom fruits and vegetables for your garden Books | Crafts, Hobbies & Home | Gardening & Landscape Design     5.0
                     Into the Storm: Two Ships, a Deadly Hurricane, and an Epic Battle for Survival            Books | Engineering & Transportation | Engineering     4.0
                                   Orca: How We Came to Know and Love the Ocean's Greatest Predator                                    Books | History | Americas     4.0
```

[Analyse 4 axes]
- Coherence thematique: recouvrement categories = 0.06
- Proximite apparente (exemples):
  * 'Color Illusions' -> 'The Espressologist' (sim=0.68)
  * 'Color Illusions' -> 'Daring to Hope: Finding God's Goodness in the Broken and the Beautiful' (sim=0.50)
  * 'Color Illusions' -> 'A Mighty Long Way: My Journey to Justice at Little Rock Central High School' (sim=0.49)
- Redondance: similarite intra liste moyenne = 0.62, max = 0.99
- Sur specialisation: concentration categorie dominante train = 0.00
- Hit rate test: 0/10
- Top categories train: Books | Mystery (27.3%), Thriller & Suspense | Thrillers & Suspense (19.2%), Books | Literature & Fiction | Genre Fiction (15.0%)
- Top categories recos: Books | Literature & Fiction | Genre Fiction (22.2%), Books | Sports & Outdoors | Biographies (11.1%), Books | Literature & Fiction | United States (11.1%)

================================================================================
U5_lecteur_specialise | user_id=AE2GIR4PXGB3EQKORWFUB3J3ZTGQ
Selection: plus de 80% des lectures dans une categorie dominante

[Historique train]
```
                                                                              titre                                   categories  rating          date
Dan the Adventurer: A Gamelit Harem Fantasy Adventure (Gold Girls and Glory Book 2) Books | Literature & Fiction | Genre Fiction     5.0 1535482828619
                                                                   Herald of Shalia Books | Literature & Fiction | Genre Fiction     5.0 1564371905994
                                  Metal Mage (Metal Mage (Completed Series) Book 1) Books | Literature & Fiction | Genre Fiction     5.0 1579975364000
                                                                       Power Mage 6 Books | Literature & Fiction | Genre Fiction     5.0 1583141892514
                             Barbarian Outcast (Princesses of the Ironbound Book 1) Books | Literature & Fiction | Genre Fiction     4.0 1583386328492
                                                                 Herald of Shalia 3 Books | Literature & Fiction | Genre Fiction     4.0 1583387423872
                                                                 Soul Gem Collector Books | Literature & Fiction | Genre Fiction     4.0 1583485352731
                                      Metal Mage 11 (Metal Mage (Completed Series)) Books | Literature & Fiction | Genre Fiction     5.0 1583993215648
  Dragon Emperor 8: From Human to Dragon to God (Dragon Emperor (Completed Series)) Books | Literature & Fiction | Genre Fiction     4.0 1586307815190
                                                                 Herald of Shalia 4 Books | Literature & Fiction | Genre Fiction     5.0 1586407210370
                                                   Scholomance: The Devil's Academy Books | Literature & Fiction | Genre Fiction     5.0 1586497628377
                                                               Soul Gem Collector 2 Books | Literature & Fiction | Genre Fiction     5.0 1586744156288
                                      Metal Mage 12 (Metal Mage (Completed Series)) Books | Literature & Fiction | Genre Fiction     5.0 1586995593859
                            Barbarian Assassin (Princesses of the Ironbound Book 2) Books | Literature & Fiction | Genre Fiction     4.0 1588313776734
                                        The Five Trials (Tsun-Tsun TzimTzum Book 1) Books | Literature & Fiction | Genre Fiction     4.0 1589009752027
                                                                    Princess Master Books | Literature & Fiction | Genre Fiction     5.0 1592121544150
 Dragon Emperor 10: From Human to Dragon to God (Dragon Emperor (Completed Series)) Books | Literature & Fiction | Genre Fiction     5.0 1593056019566
                                               Bounty Hunting: For Gold and Revenge Books | Literature & Fiction | Genre Fiction     4.0 1594164346380
                                                                  Princess Master 2 Books | Literature & Fiction | Genre Fiction     5.0 1594770794588
                                The Duelist (The Duelist (Completed Series) Book 1) Books | Literature & Fiction | Genre Fiction     5.0 1602630118945
```

[Recommandations top N]
```
 rang       asin                   titre_recommande                                   categories                score_source
    1 B003NHR9EM                 The Espressologist                                              top_n_indices_similarity.py
    2 B0BLP6VCTS                 System Overclocked Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
    3 B08V9BZHH9                         Shadowmark Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
    4 B006284PBO Fire with Fire (Demonblood Book 2) Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
    5 B07Q2FXGRH                      The Crowlands Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
    6 B074HH6QVW                          RotaryPug Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
    7 0553248642                   The Haj: A Novel Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
    8 0425197441                            Xombies Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
    9 1447286006                        Great Alone Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
   10 B00HIHUPNK                          Migrators Books | Literature & Fiction | Genre Fiction top_n_indices_similarity.py
```

[Verite terrain test]
```
                                         titre                                   categories  rating
          Arena Road: A Reverse Portal Fantasy Books | Literature & Fiction | Genre Fiction     5.0
                       Resurrected as a Drow 3 Books | Literature & Fiction | Genre Fiction     5.0
Dungeon In My Closet: A Reverse Portal Fantasy Books | Literature & Fiction | Genre Fiction     5.0
   Summoner School 2: A Reverse Portal Fantasy Books | Literature & Fiction | Genre Fiction     5.0
      Pocket Dungeon: A Reverse Portal Fantasy Books | Literature & Fiction | Genre Fiction     5.0
  Backyard Dungeon 6: A Reverse Portal Fantasy Books | Literature & Fiction | Genre Fiction     5.0
     Werewolf Knight: A Reverse Portal Fantasy Books | Literature & Fiction | Genre Fiction     5.0
    Master Class: A Slice of Life Harem LitRPG Books | Literature & Fiction | Genre Fiction     5.0
                                  Dream Master Books | Literature & Fiction | Genre Fiction     5.0
                        Fantasy World Mob Boss Books | Literature & Fiction | Genre Fiction     5.0
```

[Analyse 4 axes]
- Coherence thematique: recouvrement categories = 1.00
- Proximite apparente (exemples):
  * 'Dan the Adventurer: A Gamelit Harem Fantasy Adventure (Gold Girls and Glory Book 2)' -> 'The Espressologist' (sim=0.67)
  * 'Dan the Adventurer: A Gamelit Harem Fantasy Adventure (Gold Girls and Glory Book 2)' -> 'Fire with Fire (Demonblood Book 2)' (sim=0.57)
  * 'Dan the Adventurer: A Gamelit Harem Fantasy Adventure (Gold Girls and Glory Book 2)' -> 'System Overclocked' (sim=0.56)
- Redondance: similarite intra liste moyenne = 0.91, max = 1.00
- Sur specialisation: concentration categorie dominante train = 1.00
- Hit rate test: 0/10
- Top categories train: Books | Literature & Fiction | Genre Fiction (100.0%)
- Top categories recos: Books | Literature & Fiction | Genre Fiction (100.0%)

================================================================================
Synthese finale

[Tableau recapitulatif]
```
                   utilisateur                profil  nb_train  recouvrement  intra_moy  intra_max hit_rate  concentration  sur_specialisation_probable
AH2GI5KLJT2AKVCYXV2OIIGGLT4Q_1       U1_gros_lecteur       964         0.143      0.639      0.994     0/10          0.200                        False
  AH5GNK4DJYJLZDIEDV6EF5JRA4EQ     U2_lecteur_modere        24         0.357      0.595      0.707     0/10          0.400                        False
  AE3RDSKZEG2MRMMY32NZMEDO3XEQ      U3_petit_lecteur         3         0.500      0.607      0.951     0/10          0.111                        False
  AH665SQ6SQF6DXAGYIQFCX76LALA U4_lecteur_eclectique       786         0.059      0.622      0.994     0/10          0.000                        False
  AE2GIR4PXGB3EQKORWFUB3J3ZTGQ U5_lecteur_specialise        42         1.000      0.907      1.000     0/10          1.000                         True
```