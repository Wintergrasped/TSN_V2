document.addEventListener("DOMContentLoaded", () => {
  const refreshers = document.querySelectorAll("[data-refresh]");
  refreshers.forEach((button) => {
    button.addEventListener("click", async () => {
      const target = button.getAttribute("data-refresh");
      button.disabled = true;
      try {
        const res = await fetch(target);
        if (!res.ok) throw new Error("Refresh failed");
        const payload = await res.json();
        console.info("Refresh payload", payload);
      } catch (err) {
        console.error(err);
        alert("Unable to refresh; check logs");
      } finally {
        button.disabled = false;
      }
    });
  });
});
