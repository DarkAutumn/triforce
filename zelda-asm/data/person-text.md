# Person text

How to read this: one row per text string of [`Person_Text`](../src/bank1/cave-init-person-text.asm#L54) (`dat/PersonText.dat`), in the order of [`Person_TextAddrs`](../src/bank1/cave-init-person-text.asm#L47). `sel` is the text
selector (the value stored in [`PersonTextSelector`](../src/CaveVars.inc#L6)); `offset` and `len` locate the string in the blob. `text` is
what the textbox shows, lines separated by ` / `, with the blanks at both ends of each line removed;
`indent` is the number of leading blank cells of each line (`f` = fast blanks, code `$25`, that take no
time; `s` = ordinary blanks, code `$24`); `line` is the textbox line each part goes to. `said by` comes
from the selector tables. The encoding is in `README.md`. [`Person_UpdateState_Textbox`](../src/bank1/person-textbox.asm#L51)

- Cave persons: [`Cave_PersonTextSelectors`](../src/bank1/cave-init-person-text.asm#L71), indexed by person type - `$6A`. [`Cave_Init_SetUpTextAndWares`](../src/bank1/cave-init-person-text.asm#L145)
- Hint caves, after Link pays: [`Cave_HintTextSelectorsType75`](../src/bank1/cave-hint-money-game.asm#L42) for person type `$75`, [`Cave_HintTextSelectorsOther`](../src/bank1/cave-hint-money-game.asm#L48) for the others, indexed by the ware touched. [`Cave_State5HintOrMoneyGame`](../src/bank1/cave-hint-money-game.asm#L61)
- Dungeon persons: selector tables by level through [`Person_InitUnderworld_JumpTable`](../src/bank1/underworld-person-init.asm#L53), indexed by object type - `$4B`. [`Person_TextSelectorsLevels1257`](../src/bank1/underworld-person-init.asm#L67), [`Person_TextSelectorsLevels3468`](../src/bank1/underworld-person-init.asm#L134), [`Person_TextSelectorsLevel9`](../src/bank1/underworld-person-init.asm#L162)
- The life-or-money person and Grumble set their selector directly. [`Person_InitLifeOrMoney_Full`](../src/bank1/underworld-person-init.asm#L105), [`Grumble_Init_Full`](../src/bank1/underworld-person-init.asm#L216)

```
sel  | n   | offset  | len  | line   | indent    | text                                                                 | said by
$00  | 0   | 0       | 43   | 1,2    | 2f,4f     | IT'S DANGEROUS TO GO / ALONE! TAKE THIS.                             | cave person $6A
$02  | 1   | 43      | 42   | 1,2    | 2f,3f     | MASTER USING IT AND / YOU CAN HAVE THIS.                             | cave person $6C; cave person $6D
$04  | 2   | 85      | 24   | 1      | 1f        | TAKE ANY ROAD YOU WANT.                                              | cave person $6E
$06  | 3   | 109     | 42   | 1,2    | 1f,4f     | SECRET IS IN THE TREE / AT THE DEAD-END.                             | cave person $6F
$08  | 4   | 151     | 34   | 1,2    | 3f,3f     | LET'S PLAY MONEY / MAKING GAME.                                      | cave person $70
$0A  | 5   | 185     | 40   | 1,2    | 2f,5f     | PAY ME FOR THE DOOR / REPAIR CHARGE.                                 | cave person $71
$0C  | 6   | 225     | 37   | 1,2    | 4f,7f     | SHOW THIS TO THE / OLD WOMAN.                                        | cave person $72
$0E  | 7   | 262     | 39   | 1,2    | 4f,6f     | MEET THE OLD MAN / AT THE GRAVE.                                     | cave person $73
$10  | 8   | 301     | 30   | 1,2    | 2f,2f     | BUY MEDICINE BEFORE / YOU GO.                                        | cave person $74
$12  | 9   | 331     | 23   | 1      | 2f        | PAY ME AND I'LL TALK.                                                | cave person $75; cave person $76
$14  | 10  | 354     | 36   | 1,2    | 3f,8f     | THIS AIN'T ENOUGH / TO TALK.                                         | hint cave, person $75, ware 0; hint cave, person $75, ware 1; hint cave, other persons, ware 0
$16  | 11  | 390     | 38   | 1,2    | 7f,3f     | GO UP,UP, / THE MOUNTAIN AHEAD.                                      | hint cave, person $75, ware 2
$18  | 12  | 428     | 52   | 1,2,3  | 2f,2f,2s  | GO NORTH,WEST,SOUTH, / WEST TO THE FOREST / OF MAZE.                 | hint cave, other persons, ware 1
$1A  | 13  | 480     | 21   | 1      | 4f        | BOY, YOU'RE RICH!                                                    | hint cave, other persons, ware 2
$1C  | 14  | 501     | 23   | 1      | 1f        | BUY SOMETHIN' WILL YA!                                               | cave person $77; cave person $78
$1E  | 15  | 524     | 37   | 1,2    | 5f,3f     | BOY, THIS IS / REALLY EXPENSIVE!                                     | cave person $79; cave person $7A
$20  | 16  | 561     | 23   | 1      | 1f        | TAKE ANY ONE YOU WANT.                                               | cave person $6B
$22  | 17  | 584     | 36   | 1,2    | 5f,5f     | IT'S A SECRET / TO EVERYBODY.                                        | cave person $7B; cave person $7C; cave person $7D
$24  | 18  | 620     | 21   | 1      | 3f        | GRUMBLE,GRUMBLE...                                                   | Grumble $36
$26  | 19  | 641     | 40   | 1,2    | 2f,5f     | EASTMOST PENNINSULA / IS THE SECRET.                                 | dungeon person $4C in level 1,2,5,7
$28  | 20  | 681     | 24   | 1      | 1f        | DODONGO DISLIKES SMOKE.                                              | dungeon person $4B in level 1,2,5,7
$2A  | 21  | 705     | 64   | 1,2,3  | 1f,1f,1f  | DID YOU GET THE SWORD / FROM THE OLD MAN ON / TOP OF THE WATERFALL?  | dungeon person $4B in level 3,4,6,8
$2C  | 22  | 769     | 35   | 1,2    | 5f,7f     | WALK INTO THE / WATERFALL.                                           | dungeon person $4E in level 3,4,6,8
$2E  | 23  | 804     | 44   | 1,2    | 2f,3f     | SECRET POWER IS SAID / TO BE IN THE ARROW.                           | dungeon person $4D in level 1,2,5,7
$30  | 24  | 848     | 42   | 1,2    | 4f,1f     | DIGDOGGER HATES / CERTAIN KIND OF SOUND.                             | dungeon person $4E in level 1,2,5,7
$32  | 25  | 890     | 42   | 1,2    | 4f,3f     | I BET YOU'D LIKE / TO HAVE MORE BOMBS.                               | dungeon person $4F in level 1,2,5,7
$34  | 26  | 932     | 44   | 1,2    | 4f,1f     | IF YOU GO IN THE / DIRECTION OF THE ARROW.                           | dungeon person $52 in level 1,2,5,7
$36  | 27  | 976     | 35   | 1,2    | 4f,7f     | LEAVE YOUR LIFE / OR MONEY.                                          | life-or-money person $51
$38  | 28  | 1011    | 46   | 1,2    | 1f,3f     | THERE ARE SECRETS WHERE / FAIRIES DON'T LIVE.                        | dungeon person $4C in level 3,4,6,8
$3A  | 29  | 1057    | 35   | 1,2    | 4f,7f     | AIM AT THE EYES / OF GOHMA.                                          | dungeon person $4D in level 3,4,6,8
$3C  | 30  | 1092    | 40   | 1,2    | 2f,4f     | SOUTH OF ARROW MARK / HIDES A SECRET.                                | dungeon person $52 in level 3,4,6,8
$3E  | 31  | 1132    | 43   | 1,2    | 2f,2f     | THERE'S A SECRET IN / THE TIP OF THE NOSE.                           | dungeon person $50 in level 1,2,5,7; (table entry of type $51 in level 1,2,5,7, not used: that type sets its own text)
$40  | 32  | 1175    | 43   | 1,2    | 3f,2f     | SPECTACLE ROCK IS / AN ENTRANCE TO DEATH.                            | dungeon person $4F in level 3,4,6,8
$42  | 33  | 1218    | 24   | 1      | 0         | 10TH ENEMY HAS THE BOMB.                                             | dungeon person $50 in level 3,4,6,8; (table entry of type $51 in level 3,4,6,8, not used: that type sets its own text)
$44  | 34  | 1242    | 45   | 1,2    | 1f,1f     | ONES WHO DOES NOT HAVE / TRIFORCE CAN'T GO IN.                       | dungeon person $4B in level 9
$46  | 35  | 1287    | 21   | 1      | 3f        | PATRA HAS THE MAP.                                                   | dungeon person $4C in level 9
$48  | 36  | 1308    | 22   | 1      | 2f        | GO TO THE NEXT ROOM.                                                 | dungeon person $4D in level 9
$4A  | 37  | 1330    | 36   | 1,2    | 5f,5f     | EYES OF SKULL / HAS A SECRET.                                        | dungeon person $4E in level 9
```
