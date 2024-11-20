% Facts
parent(john, mary).
parent(john, mike).
parent(susan, mary).
parent(susan, mike).
parent(mary, kate).
parent(mary, tom).
parent(mike, anna).

% Rules
father(X, Y) :- parent(X, Y), male(X).
mother(X, Y) :- parent(X, Y), female(X).

male(john).
male(mike).
male(tom).

female(susan).
female(mary).
female(kate).
female(anna).

sibling(X, Y) :- parent(Z, X), parent(Z, Y), X \= Y.
grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
grandchild(X, Y) :- grandparent(Y, X).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

% Queries:
% ?- father(john, mary).     % Is John the father of Mary?
% ?- sibling(mary, mike).    % Are Mary and Mike siblings?
% ?- grandparent(john, kate). % Is John the grandparent of Kate?
% ?- ancestor(john, kate).   % Is John an ancestor of Kate?
