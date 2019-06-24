defmodule PizzaReservationsWithBias do
  def predict(reservation_count, weight, bias) do
    reservation_count * weight + bias
  end

  def train(reservations, pizzas, iterations, learning_rate) do
    Enum.reduce_while(1..iterations, {0, 0}, fn _each, {weight, bias} ->
      current_loss = loss(reservations, pizzas, weight, bias)

      cond do
        loss(reservations, pizzas, weight + learning_rate, bias) < current_loss ->
          {:cont, {weight + learning_rate, bias}}

        loss(reservations, pizzas, weight - learning_rate, bias) < current_loss ->
          {:cont, {weight - learning_rate, bias}}

        loss(reservations, pizzas, weight, bias + learning_rate) < current_loss ->
          {:cont, {weight, bias + learning_rate}}

        true ->
          {:halt, {Float.round(weight, 2), Float.round(bias, 2)}}
      end
    end)
  end

  defp loss(reservations, pizzas, weight, bias) do
    losses(reservations, pizzas, weight, bias)
    |> average
  end

  defp predictions(reservations, weight, bias) do
    reservations
    |> Enum.map(&predict(&1, weight, bias))
  end

  defp losses(reservations, pizzas, weight, bias) do
    predictions(reservations, weight, bias)
    |> Enum.zip(pizzas)
    |> Enum.map(fn {prediction, actual} -> (prediction - actual) |> :math.pow(2) end)
  end

  defp average(values) do
    Enum.sum(values) / length(values)
  end
end
