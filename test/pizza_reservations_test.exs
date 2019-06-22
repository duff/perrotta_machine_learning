defmodule PizzaReservationsTest do
  use ExUnit.Case

  test "predict" do
    assert PizzaReservations.predict(20, 2.1) == 42
  end

  test "train" do
    {pizzas, reservations} = parse_file()

    approximate_weight = PizzaReservations.train(pizzas, reservations, 10000, 0.01)

    assert approximate_weight == 1.84
    assert_in_delta PizzaReservations.predict(20, approximate_weight), 36.8, 0.01
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
