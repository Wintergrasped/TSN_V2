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

  const feedButton = document.getElementById("net-feed-refresh");
  const feedTable = document.querySelector("#net-control-feed tbody");
  const feedUrl = feedButton?.dataset.feedUrl;

  async function refreshFeed() {
    if (!feedUrl || !feedTable) return;
    feedButton.disabled = true;
    try {
      const res = await fetch(feedUrl);
      if (!res.ok) throw new Error("feed failed");
      const rows = await res.json();
      feedTable.innerHTML = rows
        .map(
          (row) => `
            <tr>
              <td>${row.detected_at}</td>
              <td>${row.callsign_id}</td>
              <td>${row.confidence ?? "—"}</td>
              <td>${(row.context || row.transcript || "").slice(0, 240)}</td>
            </tr>`
        )
        .join("");
    } catch (error) {
      console.error("feed_refresh_error", error);
    } finally {
      feedButton.disabled = false;
    }
  }

  if (feedButton && feedTable) {
    feedButton.addEventListener("click", (event) => {
      event.preventDefault();
      refreshFeed();
    });
    setInterval(refreshFeed, 15000);
  }
});
