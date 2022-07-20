package com.avih.set_solver.set_game;

public class SetCard {

    public enum Shape{CIRCLE, DIAMOND, SQUIGGLE}
    public enum Color{GREEN, RED, PURPLE}
    public enum Number{SINGLE, PAIR, TRIPPLE}

    private Shape shape;
    private Color color;
    private Number number;

    public SetCard(Shape shape, Color color, Number number) {
        this.shape = shape;
        this.color = color;
        this.number = number;
    }

    @Override
    public String toString() {
        return "SetCard_" +
                 shape +
                "_" + color +
                "_" + number;
    }
}
