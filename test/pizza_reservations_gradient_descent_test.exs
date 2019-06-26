defmodule PizzaReservationsGradientDescentTest do
  use ExUnit.Case, async: true

  test "predict" do
    assert PizzaReservationsGradientDescent.predict(20, 2.1, 7) == 49
  end

  test "train" do
    {pizzas, reservations} = parse_file()

    approximate_weight = PizzaReservationsGradientDescent.train(pizzas, reservations, 100, 0.001)

    assert_in_delta approximate_weight, 1.8436928702, 0.00000000001
  end

  defp parse_file do
    File.read!("test/support/data/pizza_reservations.txt")
    |> String.split("\n", trim: true)
    |> Kernel.tl()
    |> Enum.map(fn each ->
      [reservations, pizzas] = String.split(each)
      {String.to_integer(reservations), String.to_integer(pizzas)}
    end)
    |> Enum.unzip()
  end
end
