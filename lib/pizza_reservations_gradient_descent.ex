defmodule PizzaReservationsGradientDescent do
  def predict(reservation_count, weight, bias) do
    reservation_count * weight + bias
  end

  def train(reservations, pizzas, iterations, learning_rate) do
    Enum.reduce(1..iterations, 0, fn each, weight ->
      IO.puts("Iteration #{each} -> #{loss(reservations, pizzas, weight)}")
      IO.inspect(gradient(reservations, pizzas, weight), label: :gradient)
      weight - gradient(reservations, pizzas, weight) * learning_rate
    end)
  end

  defp gradient(reservations, pizzas, weight) do
    reservations
    |> Enum.zip(pizzas)
    |> Enum.map(fn {reservation_count, actual_pizza_count} ->
      2 * reservation_count * (predict(reservation_count, weight, 0) - actual_pizza_count)
    end)
    |> average
  end

  defp average(values) do
    Enum.sum(values) / length(values)
  end

  defp loss(reservations, pizzas, weight) do
    losses(reservations, pizzas, weight)
    |> average
  end

  defp predictions(reservations, weight) do
    reservations
    |> Enum.map(&predict(&1, weight, 0))
  end

  defp losses(reservations, pizzas, weight) do
    predictions(reservations, weight)
    |> Enum.zip(pizzas)
    |> Enum.map(fn {prediction, actual} -> (prediction - actual) |> :math.pow(2) end)
  end
end
