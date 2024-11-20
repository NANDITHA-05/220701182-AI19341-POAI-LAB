unification
?- X = 5.
X = 5.
?- X = Y.
X = Y.
?- parent(john, X) = parent(john, mary).
X = mary.
?- parent(john, X) = parent(susan, mary).
false.
?- parent(john, X) = parent(susan, mary).
false.

resolution 
parent(john, mary).
parent(mary, kate).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
