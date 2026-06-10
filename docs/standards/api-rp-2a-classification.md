# API RP 2A-WSD Section 4.3.1 Joint Classification

## Core Rule
Classification applies to INDIVIDUAL BRACES per LOAD CASE.
It is NOT a geometric property of the joint.

## Classification Logic

| Class | Condition                                              |
|-------|--------------------------------------------------------|
| K     | Punching load balanced by braces, same plane same side |
| T&Y   | Punching load reacted by chord beam shear              |
| Cross | Load transferred through chord to opposite side        |
| Mixed | Fraction of each: interpolate SCF                      |

## Interpolation
SCF = f_K*SCF_K + f_TY*SCF_TY + f_cross*SCF_cross
where f_K + f_TY + f_cross = 1.0
